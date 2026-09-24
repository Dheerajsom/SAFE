# ***************************************************************************
#  SAFE — shared configuration
#  --------------------------------------------------------------------------
#  Physical limits, practical-significance gates, and display names shared by
#  the health engine (safe.health, safe.profiles) and the period analysis (safe.periods).
# ***************************************************************************

# Cleaner display names for known measurements
SENSOR_DISPLAY_NAMES = {
    'IPS7100MHC001': 'IPS7100_MHC_001',
}

# IPS7100 particulate-matter size bins (all µg/m³, cumulative by size)
PM_BOUNDS = (0.0, 10000.0)
PM_METRICS = ('pm0_1', 'pm0_3', 'pm0_5', 'pm1_0', 'pm2_5', 'pm5_0', 'pm10_0')

# IPS7100 differential particle-count bins.  The deployed sensor's count
# output is particles/liter (the IPS protocol's default unit), with a stated
# measurement limit of 1,000,000 particles/liter.
PC_BOUNDS = (0.0, 1_000_000.0)
PC_METRICS = ('pc0_1', 'pc0_3', 'pc0_5', 'pc1_0', 'pc2_5', 'pc5_0', 'pc10_0')

# Hard physical bounds per metric. Values outside these are impossible for the
# instrument and are treated as failures, never as data.
HARD_BOUNDS = {
    'temperature':  (-40.0, 100.0),    # Celsius
    'humidity':     (0.0, 100.0),      # %RH
    'pressure':     (300.0, 1200.0),   # hPa
    'shuntVoltage': (-0.320, 0.320),   # INA219 max shunt voltage range (V)
    **{pm: PM_BOUNDS for pm in PM_METRICS},
    **{pc: PC_BOUNDS for pc in PC_METRICS},
}

# Device associated with the historical short display name.
SENSOR_DISPLAY_DEVICES = {'IPS7100MHC001': '001e064a1520'}

# --------------------------------------------------------------------------
# Practical-significance gates (shared by streaming + period analysis)
# --------------------------------------------------------------------------
# With large, autocorrelated samples a Welch/Levene p-value collapses toward 0
# for practically meaningless shifts. Statistical significance is therefore
# necessary but NOT sufficient: we additionally require a minimum *effect size*
# before flagging drift. Effect sizes are scale-free and do not inflate with
# sample size.
#   - MIN_COHENS_D : Cohen's "small" effect floor for a real mean shift
#                    (0.2 = small, 0.5 = medium, 0.8 = large).
#   - MIN_STD_RATIO: spread must change by >= +50% or <= -33% to count as a
#                    variance shift.
MIN_COHENS_D = 0.2
MIN_STD_RATIO = 1.5

# Smallest move of a *constant* level that counts as a step-change between two
# flat windows, per metric (units match HARD_BOUNDS).
FLAT_MEAN_SHIFT_THRESHOLDS = {
    'temperature':  0.05,   # °C
    'humidity':     0.2,    # %RH
    'pressure':     0.05,   # hPa
    'shuntVoltage': 0.001,  # V
    **{pm: 0.1 for pm in PM_METRICS},   # µg/m³
    **{pc: 10.0 for pc in PC_METRICS},  # particles/L
}
DEFAULT_FLAT_MEAN_SHIFT = 0.01

# Variance below this is considered flat (a constant signal)
FLAT_VAR_THRESHOLD = 1e-12
