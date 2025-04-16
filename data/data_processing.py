import os 
import numpy as np
import polars as pl 
import ipdb
dir = '/storage2/payal/dropbox/private/data/'
writefile = '/storage2/payal/dropbox/private/SILVER/data/data.parquet'

seed = 140799
np.random.seed(seed)
pl.set_random_seed(seed)
mrn = pl.read_parquet(os.path.join(dir, 'processed', 'mrn.parquet'))
external_cohort = mrn.filter(pl.col('mgh_mrn').is_null()).select('empi').to_numpy().reshape(-1)

split = pl.concat([
    mrn.filter(
        ~pl.col('empi').is_in(external_cohort)
    ).with_columns(
        pl.Series(
        name='split',
        values=np.random.choice(
            a=['train','tune','test'],
            size=mrn.height-len(external_cohort)-1,
            replace=True,
            p=[.75,.10,.15]
        ))
    ),
    mrn.filter(
        pl.col('empi').is_in(external_cohort)
    ).with_columns(
        pl.lit('external').alias('split')
    ), 
])

def separate_cohorts(df, external_cohort): 
    return df.filter(
        ((pl.col('empi').is_in(external_cohort)) & (pl.col('hospital')=='BWH'))
        | 
        ((~pl.col('empi').is_in(external_cohort)) & (pl.col('hospital')=='MGH'))
    ).drop(
        'hospital'
    )

def diagnosis_date(name): 
    df = pl.read_parquet(os.path.join(dir, 'processed', 'diagnosis.parquet'))

    disease_names = df.select(['name']
    ).unique().filter(
        pl.col('name').str.to_lowercase().str.contains(name)
    ).to_numpy().reshape(-1)

    return df.filter(
        pl.col('name').is_in(disease_names)
    ).select(
        ['empi', 'date']
    ).sort(
        ['empi', 'date',]
    ).unique(
        subset='empi',
        keep='first'
    ).rename(
        {'date':'diagnosis_date'}
    )

def add_comorbidity(df, name): 
    return df.join(
        diagnosis_date(name), 
        on='empi',
        how='left',
    ).with_columns(
        pl.when(
            pl.col('diagnosis_date') <= pl.col('ecg_date')
        ).then(
            1
        ).otherwise(
            0
        ).alias(
            name.replace(' ','_')
        )
    ).drop(
        'diagnosis_date'
    )

def after_hf_diagnosis(df, date_col): 
    return df.join(
        diagnosis_date('heart failure'),
        on='empi',
    ).filter(
        pl.col('diagnosis_date')<=pl.col(date_col)
    ).with_columns(
        (pl.col(date_col)-pl.col('diagnosis_date')).dt.total_days().alias('days_since_diagnosis')
    ).drop(
        'diagnosis_date'
    )

ecg = pl.read_parquet(
    os.path.join(dir, 'processed', 'ecg.parquet')
).pipe(
    separate_cohorts, external_cohort
).rename(
    {'date':'ecg_date', 'path':'ecg_path'}
).with_columns( # drop ECGs within 1 hour of each other 
    pl.col('ecg_date').dt.round(every='1h').alias('ecg_date_round')
).filter(
    pl.col('ecg_date')<=pl.date(year=2025,month=1,day=1)
).sort(
    ['empi','ecg_date']
).unique(
    subset=['empi','ecg_date_round'],
    keep='last',
).drop(
    'ecg_date_round'
).pipe(
    after_hf_diagnosis,
    date_col='ecg_date',
).pipe(
    add_comorbidity, name='diabetes mellitus'
).pipe(
    add_comorbidity, name='hypertension'
).pipe(
    add_comorbidity, name='atheroscler'
).pipe(
    add_comorbidity, name='chronic obstructive pulmonary disease'
).pipe(
    add_comorbidity, name='atrial fibrillation'
).unique(
    ['empi','ecg_date']
)

# admission
ecg = ecg.join(
    pl.read_parquet(
        os.path.join(dir, 'processed', 'encounter.parquet')
    ).pipe(
        separate_cohorts, external_cohort
    ).pipe(
        after_hf_diagnosis,
        date_col='admit_date',
    ).drop(
        'days_since_diagnosis'
    ).filter(
        pl.col('admit_date') != pl.col('discharge_date')
    ).join(
        ecg,
        on='empi',
        how='inner',
    ).filter(
        pl.col('admit_date')<=pl.col('ecg_date')
    ).filter(
        pl.col('discharge_date')<=pl.col('ecg_date')
    ).group_by(
        ['empi','ecg_date']
    ).agg(
        pl.col('admit_date').unique().len().alias('hospitalisations')
    ),
    on=['empi','ecg_date'],
    how='left',
).with_columns(
    pl.col('hospitalisations').fill_null(0)
)

ecg = ecg.join(
    pl.read_parquet(
        os.path.join(dir, 'processed', 'encounter.parquet')
    ).pipe(
        separate_cohorts, external_cohort
    ).pipe(
        after_hf_diagnosis,
        date_col='admit_date',
    ).drop(
        'days_since_diagnosis'
    ).filter(
        pl.col('admit_date') != pl.col('discharge_date')
    ).join(
        ecg,
        on='empi',
        how='inner',
    ).filter(
        (pl.col('admit_date') <= pl.col('ecg_date')) & 
        (pl.col('admit_date') >= (pl.col('ecg_date') - pl.duration(days=365)))
    ).group_by(
        ['empi','ecg_date']
    ).agg(
        pl.col('admit_date').unique().len().alias('hospitalisations_past_year')
    ),
    on=['empi','ecg_date'],
    how='left',
).with_columns(
    pl.col('hospitalisations_past_year').fill_null(0)
)

# transplant
ecg = pl.read_parquet(
    os.path.join(dir, 'processed', 'procedure.parquet')
).pipe(
    separate_cohorts, external_cohort,
).pipe(
    after_hf_diagnosis,
    date_col='date',
).drop(
    'days_since_diagnosis'
).filter(
    pl.col('name').str.to_lowercase().str.contains('transplant')
).filter(
    pl.col('name').str.to_lowercase().str.contains('heart')
).filter(
    ~pl.col('name').str.to_lowercase().str.contains('cadaver')
).drop(
    'name'
).join(
    ecg,
    on=['empi'],
    how='right',
).with_columns(
    pl.when(
        pl.col('date') < pl.col('ecg_date')
    ).then(
        1
    ).otherwise(
        0
    ).alias(
        'transplant'
    )
).drop(
    'date'
).unique(
    ['empi','ecg_date']
)

drugs = { 
    'dapagliflozin':'sglt2',
    'empagliflozin':'sglt2',
    'enalapril':'angio',
    'captopril':'angio',
    'lisinopril':'angio',
    'valsartan':'angio',
    'sacubitril':'angio',
    'losartan':'angio',
    'atenolol':'betablocker',
    'metoprolol':'betablocker',
    'carvedilol':'betablocker',
    'furosemide':'diuretic',
    'torasemide':'diuretic',
    'bumetanide':'diuretic',
    'spironolactone':'mra',
    'eplerenone':'mra',

    'ramipril':'angio',
    'benazepril':'angio',
    'fosinopril':'angio',
    'perindopril':'angio',
    'quinapril':'angio',
    'trandolapril':'angio',
    'irbesartan':'angio',
    'olmesartan':'angio',
    'telmisartan':'angio',
    'candesartan':'angio',
    'eprosartan':'angio',
    'metolazone':'diuretic',
    'etacrynic acid':'diuretic',
    'amiloride':'diuretic',
    'triamterene':'diuretic',
}

# medications
ecg = ecg.join(
    pl.read_parquet(
        os.path.join(dir, 'processed', 'medication.parquet')
    ).pipe(
        separate_cohorts, external_cohort
    ).pipe(
        after_hf_diagnosis, date_col='start_date',
    ).drop(
        'days_since_diagnosis'
    ).filter(
        pl.col('drugbank_name').is_in(drugs.keys())
    ).with_columns(
        pl.col('drugbank_name').replace_strict(drugs).alias('medication')
    ).drop(
        'drugbank_name'
    ).join(
        ecg.select(['empi','ecg_date']),
        on='empi',
        how='inner',
    ).filter(
        pl.col('start_date')<=pl.col('ecg_date') 
    ).filter(
        pl.when(
            pl.col('stop_date').is_not_null()
        ).then(
            pl.col('stop_date')>=pl.col('ecg_date')
        ).otherwise(
            True
        ) 
    ).drop(
        ['start_date','stop_date']
    ).unique(
    ).group_by(
        ['empi','ecg_date']
    ).agg([
        pl.col('medication').n_unique().alias('num_meds'),
        (pl.col('medication')=='slgt2').any().alias('slgt2'),
        (pl.col('medication')=='angio').any().alias('angio'),
        (pl.col('medication')=='betablocker').any().alias('betablocker'),
        (pl.col('medication')=='mra').any().alias('mra'),
        (pl.col('medication')=='diuretic').any().alias('diuretic'),
    ]),
    on=['empi','ecg_date'],
    how='left',
).with_columns(
    pl.col('num_meds').fill_null(0),
    pl.col('slgt2').fill_null(False),
    pl.col('angio').fill_null(False),
    pl.col('betablocker').fill_null(False),
    pl.col('mra').fill_null(False),
    pl.col('diuretic').fill_null(False),
)

lvef = pl.read_parquet(
    os.path.join(dir, 'processed', 'lvef.parquet')
).pipe(
    separate_cohorts, external_cohort
).pipe(
    after_hf_diagnosis,
    date_col='lvef_date',
).drop(
    'days_since_diagnosis'
)

tags = ecg.select(
    ['empi','ecg_date'],
).join(
    lvef, 
    on='empi',
).filter(
    pl.col("lvef_date") < pl.col("ecg_date").dt.date()
).unique(
).sort(
    ['empi','ecg_date','lvef_date']
).group_by(
    ['empi','ecg_date']
).agg(
    pl.col("lvef").alias('lvef_history')
).with_columns(
    (pl.col('lvef_history').list.last()<40).alias('tag_hfref'),
    (pl.col('lvef_history').list.last()>50).alias('tag_50'),
    pl.col('lvef_history').list.eval(pl.element()>40).list.all().alias('tag_not_hfref'),
    pl.col('lvef_history').list.eval(pl.element()>50).list.all().alias('tag_hfpef'),
    ((pl.col('lvef_history').list.last()>40) & (pl.col('lvef_history').list.eval(pl.element()<40).list.any())).alias('tag_hfimpef'),
).drop(
    'lvef_history'
)

initial_lvef = lvef.sort(
    ['empi', 'lvef_date']
).unique(
    subset='empi', 
    keep='first',
).with_columns(
    pl.col('lvef').alias('initial_lvef'),
).select(
    pl.col('empi'),
    pl.col('initial_lvef'),
)

preceding_lvef = ecg.join(
    lvef, 
    on='empi',
).with_columns(
    (pl.col('lvef_date')-pl.col('ecg_date').dt.date()).alias('delta')
).filter(
    pl.col('delta')<0,
).sort(
    ['empi','ecg_date','delta']
).unique(
    subset=['empi','ecg_date'],
    keep='last',
).sort(
    ['delta']
).select(
    pl.col('empi'),
    pl.col('ecg_date'),
    pl.col('lvef').alias('preceding_lvef'),
    (pl.col('lvef')<=40).alias('preceding_lvef_below_40'),
    (pl.col('lvef')>40).alias('preceding_lvef_above_40'),
    (pl.col('lvef')>50).alias('preceding_lvef_above_50'),
)

prior_lvef = ecg.join(
    lvef, 
    on='empi',
).with_columns(
    (pl.col('lvef_date')-pl.col('ecg_date').dt.date()).alias('delta')
).filter(
    pl.col('delta')<pl.duration(days=-1) # prior_lvef_days_min
).filter(
    pl.col('delta')>=pl.duration(days=-365) # prior_lvef_days_max
).unique(
).sort(
    ['empi', 'ecg_date', 'lvef_date']
).group_by_dynamic(
    index_column=pl.col('ecg_date').dt.date().alias('ecg_date_tmp'),
    every='1d',
    period='365d',
    offset='-365d',
    include_boundaries=False,
    group_by=['empi', 'ecg_date'],
    start_by='datapoint',
).agg(
    pl.col('lvef').count().alias('prior_lvef_support'),
    pl.col('lvef').mean().alias('prior_lvef_mean'),
    pl.col('lvef').min().alias('prior_lvef_min'),
    pl.col('lvef').max().alias('prior_lvef_max'),
    pl.col('lvef').std().alias('prior_lvef_std').fill_null(0),
    (pl.col('lvef').max()-pl.col('lvef').min()).alias('prior_lvef_range'),
    (pl.col('lvef') <= 40).any().alias('prior_lvef_any_hfref'),
    (pl.col('lvef') <= 40).all().alias('prior_lvef_all_hfref'),
    (pl.col('lvef') >= 50).any().alias('prior_lvef_any_hfpef'),
    (pl.col('lvef') >= 50).all().alias('prior_lvef_all_hfpef'),
    ((pl.col('lvef') > 40) & (pl.col('lvef') < 50)).any().alias('prior_lvef_any_hfmref'),
    ((pl.col('lvef') > 40) & (pl.col('lvef') < 50)).all().alias('prior_lvef_all_hfmref'),
).drop(
    'ecg_date_tmp'
).unique(
    subset=['empi','ecg_date']
)

df = split.join(
    ecg,
    on='empi',
    how='inner',
).join(
    initial_lvef,
    on=['empi'],
    how='left',
).join(
    prior_lvef,
    on=['empi','ecg_date'],
    how='left',
).join(
    preceding_lvef,
    on=['empi','ecg_date'],
    how='left',
).join(
    tags,
    on=['empi','ecg_date'],
    how='left',
).with_columns(
    pl.col('tag_hfref').fill_null(False),
    pl.col('tag_not_hfref').fill_null(False),
    pl.col('tag_hfpef').fill_null(False),
    pl.col('tag_hfimpef').fill_null(False),
)

future_windows = [(1,365)] 
for (t1, t2) in future_windows: 
    future_lvef = ecg.join(
        lvef, 
        on='empi',
    ).with_columns(
        (pl.col('lvef_date')-pl.col('ecg_date').dt.date()).alias('delta')
    ).filter(
        pl.col('delta')>pl.duration(days=t1)
    ).filter(
        pl.col('delta')<=pl.duration(days=t2)
    ).unique(
    ).sort(
        ['empi','ecg_date','lvef_date']
    ).group_by_dynamic(
        index_column=pl.col('ecg_date').dt.date().alias('ecg_date_tmp'),
        every='1d',
        period=f'{t2-t1+1}d',
        include_boundaries=False,
        by=['empi','ecg_date'],
    ).agg(
        # pl.col('delta').alias('future_lvef_delta'),
        # pl.col('lvef').alias('future_lvef_values'),
        # pl.col('lvef').count().alias(f'future_{t1}_{xt2}_support'),
        # pl.col('lvef').mean().alias(f'future_{t1}_{t2}_mean'),
        # pl.col('lvef').std().alias(f'future_{t1}_{t2}_std').fill_null(0),
        # pl.col('lvef').max().alias(f'future_{t1}_{t2}_max'),
        # pl.col('lvef').min().alias(f'future_{t1}_{t2}_min'),
        # (pl.col('lvef').max()-pl.col('lvef').min()).alias(f'future_{t1}_{t2}_range'),
        (pl.col('lvef') <= 40).any().alias(f'future_{t1}_{t2}_any_below_40'),
        # (pl.col('lvef') <= 40).all().alias(f'future_{t1}_{t2}_all_below_40'),
    ).drop(
        'ecg_date_tmp'
    ).unique(
        subset=['empi','ecg_date']
    )

    df = df.join(
        future_lvef,
        on=['empi','ecg_date'],
        how='left',
    )

# mean annual hospitalizations 
df = df.with_columns(
    (pl.when(pl.col("days_since_diagnosis") > 0)
       .then(pl.col("hospitalisations") / (pl.col("days_since_diagnosis") / 365))
       .otherwise(0))
    .alias("mean_annual_hospitalizations")
)

# demographics 
df = df.join(
    pl.read_parquet(os.path.join(dir, 'processed', 'demographic.parquet')
    ).select(['empi','sex','date_of_birth']
    ).with_columns(
        (pl.col("sex") == "Female").cast(pl.Int8()).alias("sex"),
        pl.col("date_of_birth").cast(pl.Date).alias("date_of_birth"),
    ).unique(),
    on='empi'
).with_columns(
    ((pl.col('ecg_date')-pl.col('date_of_birth')).dt.total_days() / 365.25).round().alias('age')
).drop(
    'date_of_birth'
)

df.write_parquet(writefile)
print(f'Wrote dataset to {writefile}')
print(df.shape)
print(df)
print(df.columns)