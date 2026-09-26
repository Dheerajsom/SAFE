# ***************************************************************************
#  SAFE — shared configuration
#  --------------------------------------------------------------------------
#  Physical limits, practical-significance gates, and display names shared by
#  the health engine (safe.health, safe.profiles) and the period analysis (safe.periods).
# ***************************************************************************

# Health engine release. Detection changes must bump it: saved states refuse
# to load across versions, and every incident records the version it came from.
ENGINE_VERSION = "3.1.0"

# Cleaner display names for known measurements
SENSOR_DISPLAY_NAMES = {
    'IPS7100MHC001': 'IPS7100_MHC_001',
}

# IPS7100 particulate-matter size bins (all µg/m³, cumulative by size)
PM_BOUNDS = (0.0, 10000.0)
PM_METRICS = ('pm0_1', 'pm0_3', 'pm0_5', 'pm1_0', 'pm2_5', 'pm5_0', 'pm10_0')

# IPS7100 differential particle-count bins, in particles/liter.
PC_METRICS = ('pc0_1', 'pc0_3', 'pc0_5', 'pc1_0', 'pc2_5', 'pc5_0', 'pc10_0')

# The IPS7100 reports each PM bin as the running sum of count bins times a fixed
# mass per particle (ug/m3 per particle/L). Recovered from the valo_node_01 1 s
# export: PM matches the counts to within 0.5 % for the four smallest bins.
IPS7100_MASS_PER_COUNT = (8.3557e-07, 2.2560e-05, 1.0445e-04, 8.3642e-04, 1.3691e-02, 1.8456e-01, 2.5115e+00)

# A count bin is physically impossible once its mass alone would exceed the PM
# hard bound. The old flat 1,000,000/L limit was exceeded by real pc0_1 readings
# (up to 1.2M/L at 1 s) while PM2.5 read only 37-110 ug/m3. pc0_1's bound is also the
# display fallback for unconfigured count fields.
PC_HARD_BOUNDS = {pc: (0.0, float(f"{PM_BOUNDS[1] / mass:.3g}"))
                  for pc, mass in zip(PC_METRICS, IPS7100_MASS_PER_COUNT)}
PC_BOUNDS = PC_HARD_BOUNDS['pc0_1']

# At 1 s the IPS7100 reports whole particle counts per liter, so counting noise on
# a value averaged over T seconds has variance of about expected / T.
PC_LITERS_PER_SECOND = 1.0

# Hard physical bounds per metric. Values outside these are impossible for the
# instrument and are treated as failures, never as data.
HARD_BOUNDS = {
    'temperature':  (-40.0, 100.0),    # Celsius
    'humidity':     (0.0, 100.0),      # %RH
    'pressure':     (300.0, 1200.0),   # hPa
    'shuntVoltage': (-0.320, 0.320),   # INA219 max shunt voltage range (V)
    **{pm: PM_BOUNDS for pm in PM_METRICS},
    **PC_HARD_BOUNDS,
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


# --------------------------------------------------------------------------
# SensorHealth detector heuristics
# --------------------------------------------------------------------------
# Fixed engine logic rather than per-metric profile settings: changing any of
# these changes detection behavior and requires an ENGINE_VERSION bump.

# Fraction of a zero run's readings that must contradict clean air for a freeze.
STUCK_ZERO_SUPPORT = 0.8
# Shortest zero run that can hold shift/drift detection as a pending stuck-at-zero.
STUCK_ZERO_MIN_RUN = 2
# A value this many freeze tolerances from the freeze anchor proves the series varies.
VARIABILITY_TOLERANCE_FACTOR = 4

# Cadence: median of the last CADENCE_HISTORY intervals, once CADENCE_MIN_INTERVALS
# exist, degrades when it exceeds CADENCE_DEGRADATION_FACTOR x the expected interval
# and the slow cadence plus that history spans the profile's min_gap_seconds.
CADENCE_HISTORY = 12
CADENCE_MIN_INTERVALS = 11
CADENCE_DEGRADATION_FACTOR = 1.5

# Consecutive invalid readings that restart an unfinished warmup.
INVALID_RUN_RESTART = 3

# Provisional warmup gate: once PROVISIONAL_MIN_SAMPLES exist, a reading further
# than max(PROVISIONAL_OUTLIER_SCALES x robust scale, PROVISIONAL_OUTLIER_STEPS x
# step_min_effect) from the last PROVISIONAL_WINDOW warmup values is contamination.
# The window spans PROVISIONAL_WINDOW_SECONDS (24 readings at the 300 s cadence),
# sampled down to PROVISIONAL_WINDOW values at faster cadences: 24 readings at 1 s
# is only 24 s of history, too short to know how much ambient air moves.
PROVISIONAL_MIN_SAMPLES = 8
PROVISIONAL_WINDOW = 24
PROVISIONAL_WINDOW_SECONDS = 7200
PROVISIONAL_OUTLIER_SCALES = 12
PROVISIONAL_OUTLIER_STEPS = 4

# Windowed tests need at least this many readings per half-window (or half of
# minimum_samples, whichever is larger); a noise change needs this std ratio.
WINDOW_MIN_SAMPLES = 8
NOISE_STD_RATIO = 2

# Excess of residual anomalies in a window: both signs at least
# ANOMALY_EXCESS_MIN_EACH_SIGN, and in total at least max(ANOMALY_EXCESS_MIN,
# ANOMALY_EXCESS_FRACTION x window readings).
ANOMALY_EXCESS_MIN_EACH_SIGN = 2
ANOMALY_EXCESS_MIN = 4
ANOMALY_EXCESS_FRACTION = 0.1

# Evidence-strength heuristics attached to incidents; never calibrated probabilities.
CONFIDENCE_DEFINITE = 1.0      # physically impossible value
CONFIDENCE_OBSERVED = 0.99     # directly observed timing fault (silence, clock)
CONFIDENCE_RULE = 0.95         # rule violation or confirmed freeze
CONFIDENCE_PERSISTENT = 0.85   # persistent residual shift
CONFIDENCE_DEFAULT = 0.8       # availability degradation, reference drift
CONFIDENCE_STATISTICAL = 0.7   # windowed or cumulative statistical evidence
CONFIDENCE_WEAK = 0.6          # possible restart, startup contamination
CONFIDENCE_ISOLATED = 0.55     # single residual anomaly
