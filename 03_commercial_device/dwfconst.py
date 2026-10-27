"""
Subset of WaveForms SDK constants (excerpted from Digilent's official dwfconstants.py).

They are copied here rather than importing the official file so that the package is self-contained and
readable, with each constant's meaning explained in a comment. To use the official version:

  Windows: C:\\Program Files (x86)\\Digilent\\WaveFormsSDK\\samples\\py\\dwfconstants.py
  macOS:   /Applications/WaveForms.app/Contents/Resources/SDK/samples/py/dwfconstants.py
  Linux:   /usr/share/digilent/waveforms/samples/py/dwfconstants.py

Constants in the official file are ctypes objects (e.g. c_int(1)); here they are plain ints,
and the wrapper functions in the driver do the c_int/c_double conversion.
"""

# ---------------------------------------------------------------- device state
DwfStateReady = 0
DwfStateConfig = 4
DwfStatePrefill = 5
DwfStateArmed = 1
DwfStateWait = 7
DwfStateTriggered = 3
DwfStateRunning = 3
DwfStateDone = 2

# ------------------------------------------------------------ AnalogOut nodes
AnalogOutNodeCarrier = 0
AnalogOutNodeFM = 1
AnalogOutNodeAM = 2

# ------------------------------------------------------------ waveform function IDs
funcDC = 0
funcSine = 1
funcSquare = 2
funcTriangle = 3
funcRampUp = 4
funcRampDown = 5
funcNoise = 6
funcPulse = 7
funcTrapezium = 8
funcSinePower = 9
funcCustom = 30
funcPlay = 31

# --------------------------------------------------- AnalogOut idle state
# Disable = output high impedance; Offset = hold offset voltage; Initial = hold first waveform value
DwfAnalogOutIdleDisable = 0
DwfAnalogOutIdleOffset = 1
DwfAnalogOutIdleInitial = 2

# ------------------------------------------------------------ AnalogIn acquisition mode
acqmodeSingle = 0
acqmodeScanShift = 1
acqmodeScanScreen = 2
acqmodeRecord = 3
acqmodeOvers = 4
acqmodeSingle1 = 5

# ---------------------------------------------------------------- trigger source
trigsrcNone = 0
trigsrcPC = 1
trigsrcDetectorAnalogIn = 2
trigsrcDetectorDigitalIn = 3
trigsrcAnalogIn = 4
trigsrcDigitalIn = 5
trigsrcDigitalOut = 6
trigsrcAnalogOut1 = 7
trigsrcAnalogOut2 = 8
trigsrcAnalogOut3 = 9
trigsrcAnalogOut4 = 10
trigsrcExternal1 = 11

# ---------------------------------------------------------------- filter
filterDecimate = 0
filterAverage = 1
filterMinMax = 2

# ------------------------------------------------------------ channel numbers (0-based)
# Note: DWF.WAVEFORM_CHANNEL_1 == 0 in the Java project,
# i.e. W1 on the panel; OSCILLOSCOPE_CHANNEL_1 == 0, i.e. 1+ on the panel.
CH_W1 = 0
CH_W2 = 1
CH_SCOPE1 = 0  # panel 1+
CH_SCOPE2 = 1  # panel 2+
