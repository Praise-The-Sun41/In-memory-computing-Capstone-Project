"""
knowm_md.py — Python driver layer for the Knowm Memristor Discovery (V2 board).

Design goal: rewrite the layer of the Java project org.knowm.memristor.discovery
(DWFProxy + PulseUtility + WaveformUtils) directly in Python with ctypes,
bypassing Java / waveforms4j entirely and depending only on Digilent's dwf dynamic library.

========================= V2 board circuit (important!) =========================

The V2 board is "upside down" relative to V0/V1 (see manual Figure 18 D, MD_V2_Board.java):

        W1 (2+)  ──[ Rs ]──┬── node2 (1+) ──[ Rm ]── GND
                           │
                        C_par ≈ 180 pF
                           │
                          GND        (the 1+ probe also has 1 MΩ input impedance to ground)

  * scope 2+  measures the W1 source voltage
  * scope 1+  measures the top of the memristor (midpoint of Rs and Rm)
  * the bottom of the memristor is grounded

Therefore:
      V_memristor = V(1+)
      I           = ( V(2+) - V(1+) ) / Rs
      R_memristor = V(1+) / I

To make curve polarity match V0/V1, the Knowm GUI does two things:
      1) negates the W1 amplitude   ( startWave(..., -amplitude, -offset, ...) )
      2) negates the measured voltage and current as a whole
This library follows the same convention: the voltages you pass as `v` are in "memristor coordinates",
negated internally for the V2 board; the returned V/I are also converted back to memristor coordinates.

========================= DIO mapping =========================

V2 board: DIO 0..15 → 16 bidirectional analog switches, the 16 devices of the 1×16 linear array chip.
          (Java: PulseUtility loops i=0..15; DWFProxy sets all to 0 at power-up)
          Output enable mask = 0xFFFF.
          In Mode 2 differential mode, branch A uses DIO 0..7 and branch B DIO 8..15
          (see KTRAM_Controller_12.readSwitchStates: switchA<=7, switchB>=8)

V1 board: low 8 bits = 8 switches; high 8 bits = 4 1:4 MUXes,
          bit order [W2, W1, 2+, 1+], 2 bits each: 00=OUT, 01=A, 11=B, 10=Y
          (see MuxController.java)

========================= Safety =========================

* The AD2's V+ / V- must be on (±5 V), otherwise the analog switches do not conduct.
* Use reads of order 0.1 V; write ~1 V; erase ~-2 V (typical for Knowm SDC devices;
  follow your device's datasheet).
* Never measure a memristor with a multimeter -- the multimeter's test current rewrites the device state.
* Set the AnalogOut idle state to Idle=Offset(0 V) rather than high impedance, to avoid transients from a floating node.
"""

from __future__ import annotations

import ctypes
import math
import platform
import time

import dwfconst as C

__all__ = ["MemristorDiscovery", "DWFError", "load_dwf"]


class DWFError(RuntimeError):
    pass


# ---------------------------------------------------------------------------
# dynamic library loading
# ---------------------------------------------------------------------------
def load_dwf(path: str | None = None) -> ctypes.CDLL:
    """Load Digilent's dwf dynamic library."""
    if path:
        return ctypes.cdll.LoadLibrary(path)

    system = platform.system()
    if system == "Windows":
        # after installing WaveForms, dwf.dll is on PATH (C:\Windows\System32\dwf.dll)
        return ctypes.cdll.dwf
    if system == "Darwin":
        return ctypes.cdll.LoadLibrary("/Library/Frameworks/dwf.framework/dwf")
    return ctypes.cdll.LoadLibrary("libdwf.so")


# ---------------------------------------------------------------------------
# main class
# ---------------------------------------------------------------------------
class MemristorDiscovery:
    """
    Usage::

        with MemristorDiscovery(series_resistance=10_000) as md:
            md.select(0)                       # connect device 1
            t, vm, i = md.sweep(amplitude=1.0, frequency=100)
            r = md.read_resistance(0)

    Parameters
    ----------
    series_resistance : float
        Actual series resistance in Ω. In Mode 1 of the V2 board the two 20 kΩ at A/B in parallel = 10 kΩ.
        If you remove one, it is 20 kΩ. **This value must match the actual board**,
        otherwise all currents/resistances are wrong.
    board_version : int
        0 / 1 / 2. Default 2.
    parasitic_capacitance : float
        Parasitic node-to-ground capacitance; the Java project uses 180 pF for V2.
    """

    # from ExperimentPreferences.java
    TOTAL_PARASITIC_RESISTANCE = 160.0  # Ω, equivalent total drop of closed switch + traces
    R_SWITCH = 75.0  # Ω, on-resistance of a single bidirectional switch
    SCOPE_INPUT_R = 1e6  # Ω, input impedance of the AD2 1X probe

    MAX_BUFFER = 8192  # maximum AD2 sample buffer per channel
    # noise/offset residual of readings. Measured after calibration at about 0.4~0.8 mV.
    # it sets the **resolution limit** of resistance readings: Rs·(|w1|/thr − 1).
    READ_NOISE_V = 4e-4

    def __init__(
        self,
        series_resistance: float = 10_000.0,
        board_version: int = 2,
        parasitic_capacitance: float = 180e-12,
        dwf_path: str | None = None,
        device_index: int = -1,
        verbose: bool = True,
    ):
        self.verbose = verbose
        self.dwf = load_dwf(dwf_path)
        self.hdwf = ctypes.c_int(0)
        self.rs = float(series_resistance)
        self.board_version = int(board_version)
        self.cpar = float(parasitic_capacitance)
        self.device_index = device_index
        self._dio_state = 0
        self._opened = False
        # ---- device protection ----
        # this is the **only write outlet**; pulses/sweeps from every script end up here.
        # the protection logic itself moved to safety.SafetyGate: this place only ensures "no gate, no voltage".
        #
        # there used to be a `self.enforce_limits = True` here; a caller setting it to False
        # could bypass all clipping. **That switch has been removed**; do not add it back.
        # the only switch is "is there a gate", and without a gate the result is refusal, not permission.
        self.gate = None
        # the readings of these devices are only **lower bounds** (drop below the noise threshold, value not resolvable).
        # they are acceptable for safety (far above the floor), but do not use the number for fine inferences.
        self.read_bounded = set()
        self._track = {}        # DIO bit → device number on this path (the same number on V2)

    # ------------------------------------------------------------------ life cycle
    def __enter__(self):
        self.open()
        return self

    def __exit__(self, *exc):
        self.close()
        return False

    def _err(self, msg: str):
        buf = ctypes.create_string_buffer(512)
        self.dwf.FDwfGetLastErrorMsg(buf)
        raise DWFError(f"{msg}: {buf.value.decode(errors='replace').strip()}")

    def open(self):
        """Open the device and do the same initialisation as DWFProxy.AD2StartupWorker."""
        if not self.dwf.FDwfDeviceOpen(
            ctypes.c_int(self.device_index), ctypes.byref(self.hdwf)
        ):
            self._err("FDwfDeviceOpen failed (AD2 not connected properly? in use by the WaveForms GUI?)")
        self._opened = True

        # --- digital IO: all 16 bits as outputs, initially 0 (all switches open)
        self.dwf.FDwfDigitalIOOutputEnableSet(self.hdwf, ctypes.c_uint(0xFFFF))
        self.set_dio(0)

        # --- analog supply ±5 V (power for the analog switches/MUX; must be on)
        self.set_power(True, 5.0, -5.0)

        # --- enable both scope channels, range ±2.5 V (Java default)
        for ch in (C.CH_SCOPE1, C.CH_SCOPE2):
            self.dwf.FDwfAnalogInChannelEnableSet(self.hdwf, ctypes.c_int(ch), ctypes.c_int(1))
            self.dwf.FDwfAnalogInChannelRangeSet(self.hdwf, ctypes.c_int(ch), ctypes.c_double(2.5))
            # 1X probe, attenuation 1
            self.dwf.FDwfAnalogInChannelAttenuationSet(
                self.hdwf, ctypes.c_int(ch), ctypes.c_double(1.0)
            )

        # --- zero the waveform generator offset
        for ch in (C.CH_W1, C.CH_W2):
            self.dwf.FDwfAnalogOutNodeOffsetSet(
                self.hdwf, ctypes.c_int(ch), ctypes.c_int(C.AnalogOutNodeCarrier),
                ctypes.c_double(0.0),
            )
            # hold 0 V when stopped rather than high impedance, to avoid a floating node
            self.dwf.FDwfAnalogOutIdleSet(
                self.hdwf, ctypes.c_int(ch), ctypes.c_int(C.DwfAnalogOutIdleOffset)
            )

        # turn off auto-configure last -- the same order as Java's AD2StartupWorker.
        # afterwards every group of *Set* calls must be followed explicitly by a *Configure* to take effect.
        self.dwf.FDwfDeviceAutoConfigureSet(self.hdwf, ctypes.c_int(0))

        time.sleep(0.3)  # wait for the ±5 V supply to come up

        vp, ip, vn, inn = self.power_status()
        if vp < 4.0 or vn > -4.0:
            print(f"Warning: ±5 V supply not established (V+={vp:+.2f} V, V−={vn:+.2f} V).\n"
                  f"      The analog switches on the board will not conduct and all devices will read as open.\n"
                  f"      Check the USB supply (use a short cable, plug directly into a motherboard USB port, avoid passive hubs).")
        elif self.verbose:
            print(f"Supply ready: V+ = {vp:+.2f} V ({ip*1000:.0f} mA), "
                  f"V− = {vn:+.2f} V ({inn*1000:.0f} mA)")
        return self

    def close(self):
        """Safe power-down: open all switches, stop waveforms, turn off supplies, close the device."""
        if not self._opened:
            return
        try:
            self.stop_wave(C.CH_W1)
            self.stop_wave(C.CH_W2)
            self.dwf.FDwfAnalogInConfigure(self.hdwf, ctypes.c_int(0), ctypes.c_int(0))
            self.set_dio(0)
            self.set_power(False, 0.0, 0.0)
        finally:
            self.dwf.FDwfDeviceCloseAll()
            self._opened = False

    # ------------------------------------------------------------------ power supply
    def set_power(self, enable: bool, vpos: float = 5.0, vneg: float = -5.0):
        """
        AD2 programmable V+ / V− supplies. channel 0 = V+, channel 1 = V−; node 0 = enable, node 1 = voltage.

        Note the final `FDwfAnalogIOConfigure` -- with AutoConfigure off,
        AnalogIO, like DigitalIO, must be Configured explicitly to reach the hardware.
        Without it the supply never turns on and none of the analog switches on the board conduct,
        so the result is "the device seems open but nothing can be measured".
        """
        h = self.hdwf
        en = ctypes.c_double(1.0 if enable else 0.0)
        self.dwf.FDwfAnalogIOChannelNodeSet(h, ctypes.c_int(0), ctypes.c_int(0), en)
        self.dwf.FDwfAnalogIOChannelNodeSet(h, ctypes.c_int(0), ctypes.c_int(1),
                                            ctypes.c_double(vpos))
        self.dwf.FDwfAnalogIOChannelNodeSet(h, ctypes.c_int(1), ctypes.c_int(0), en)
        self.dwf.FDwfAnalogIOChannelNodeSet(h, ctypes.c_int(1), ctypes.c_int(1),
                                            ctypes.c_double(vneg))
        self.dwf.FDwfAnalogIOEnableSet(h, ctypes.c_int(1 if enable else 0))
        self.dwf.FDwfAnalogIOConfigure(h)          # ← key

    def power_status(self):
        """
        Read back the actual output voltage and current of V+ / V−. Returns (v_pos, i_pos, v_neg, i_neg).

        The AD2's ±5 V supply comes from USB. If the USB supply is insufficient (passive hub, poor cable),
        the voltages read back here are clearly below the set values -- the analog switches on the board half-conduct,
        showing up as drifting or uniformly high resistance readings.
        """
        h = self.hdwf
        self.dwf.FDwfAnalogIOStatus(h)
        out = []
        for ch in (0, 1):
            for node in (1, 2):     # node 1 = voltage, node 2 = current
                v = ctypes.c_double()
                self.dwf.FDwfAnalogIOChannelNodeStatus(
                    h, ctypes.c_int(ch), ctypes.c_int(node), ctypes.byref(v))
                out.append(v.value)
        return tuple(out)

    # ------------------------------------------------------------------ switch selection
    @property
    def dio(self) -> int:
        return self._dio_state

    def set_dio(self, mask: int):
        """Write the 16-bit DIO state directly."""
        self._dio_state = mask & 0xFFFF
        self.dwf.FDwfDigitalIOOutputSet(self.hdwf, ctypes.c_uint(self._dio_state))
        self.dwf.FDwfDigitalIOConfigure(self.hdwf)

    def read_dio(self) -> int:
        """Read back DIO (Java's dwf.getDigitalIOStatus())."""
        self.dwf.FDwfDigitalIOStatus(self.hdwf)
        v = ctypes.c_uint32()
        self.dwf.FDwfDigitalIOInputStatus(self.hdwf, ctypes.byref(v))
        return v.value & 0xFFFF

    def select(self, *indices: int, settle: float = 0.005):
        """
        Connect only the specified devices (0-based), disconnecting all others.
        V2: 0..15 correspond to devices 1..16 of the chip.
        """
        n = 16 if self.board_version == 2 else 8
        mask = 0
        for i in indices:
            if not 0 <= i < n:
                raise ValueError(f"device index {i} out of range (0..{n-1})")
            mask |= 1 << i
        self.set_dio(mask)
        time.sleep(settle)  # let the switch charge-injection transient decay

    def set_switch(self, index: int, on: bool, settle: float = 0.005):
        """Switch one path on/off without affecting the others (update2DigitalIOStatesAtOnce)."""
        if on:
            self.set_dio(self._dio_state | (1 << index))
        else:
            self.set_dio(self._dio_state & ~(1 << index))
        time.sleep(settle)

    def all_off(self):
        self.set_dio(0)

    # ---------------------------------------------- V1 board MUX (not used on V2)
    def set_mux(self, w1="A", w2="OUT", scope1="A", scope2="B"):
        """
        V1 board only: configure the 4 1:4 analog MUXes.
        Destination values 'A' / 'B' / 'Y' / 'OUT'. Bit order W2,W1,2+,1+, in the high 8 bits.
        """
        code = {"OUT": "00", "A": "01", "Y": "10", "B": "11"}
        bits = code[w2] + code[w1] + code[scope2] + code[scope1] + "00000000"
        upper = int(bits, 2)
        self.set_dio((self._dio_state & 0x00FF) | (upper & 0xFF00))

    # ------------------------------------------------------------------ waveforms
    def _polarity(self) -> float:
        """Polarity inversion factor of the V2 board."""
        return -1.0 if self.board_version == 2 else 1.0

    def start_wave(self, waveform: str, frequency: float, amplitude: float,
                   offset: float = 0.0, symmetry: float = 50.0, channel: int = C.CH_W1):
        """
        Start a continuous standard waveform (Java's DWF.startWave).
        amplitude / offset are in memristor coordinates; polarity is inverted internally according to the board version.
        """
        # token: no waveform may be issued without a freshly issued gate authorisation.
        # this closes the path of "calling start_wave directly to bypass clipping".
        self._require_gate("start_wave").consume_token(
            abs(amplitude) + abs(offset), "start_wave")

        func = {
            "dc": C.funcDC, "sine": C.funcSine, "square": C.funcSquare,
            "triangle": C.funcTriangle, "rampup": C.funcRampUp,
            "rampdown": C.funcRampDown, "trapezium": C.funcTrapezium,
        }[waveform.lower()]

        p = self._polarity()
        h, ch = self.hdwf, ctypes.c_int(channel)
        node = ctypes.c_int(C.AnalogOutNodeCarrier)

        self.dwf.FDwfAnalogOutNodeEnableSet(h, ch, node, ctypes.c_int(1))
        self.dwf.FDwfAnalogOutNodeFunctionSet(h, ch, node, ctypes.c_ubyte(func))
        self.dwf.FDwfAnalogOutNodeFrequencySet(h, ch, node, ctypes.c_double(frequency))
        self.dwf.FDwfAnalogOutNodeAmplitudeSet(h, ch, node, ctypes.c_double(p * amplitude))
        self.dwf.FDwfAnalogOutNodeOffsetSet(h, ch, node, ctypes.c_double(p * offset))
        self.dwf.FDwfAnalogOutNodeSymmetrySet(h, ch, node, ctypes.c_double(symmetry))
        self.dwf.FDwfAnalogOutIdleSet(h, ch, ctypes.c_int(C.DwfAnalogOutIdleOffset))
        self.dwf.FDwfAnalogOutRunSet(h, ch, ctypes.c_double(0.0))      # 0 = continuous
        self.dwf.FDwfAnalogOutRepeatSet(h, ch, ctypes.c_int(0))
        self.dwf.FDwfAnalogOutConfigure(h, ch, ctypes.c_int(1))

    def start_custom(self, samples, frequency: float, n_periods: int = 1,
                     channel: int = C.CH_W1, wait: float = 0.0):
        """
        Play a custom waveform (startCustomPulseTrain).

        samples    : voltage sequence of one period (memristor coordinates, V)
        frequency  : 1/period. For example, pulse width W and 50% duty give frequency = 1/(2W)
        n_periods  : number of periods to repeat (number of pulses)
        """
        p = self._polarity()
        data = [p * float(v) for v in samples]
        peak = max(abs(v) for v in data) or 1.0
        # token: checked against the **peak of the actual waveform**, not the amplitude the caller claims.
        # a waveform modified after authorisation (e.g. swapped for a higher shape) is also caught here.
        self._require_gate("start_custom").consume_token(peak, "start_custom")
        norm = [v / peak for v in data]

        arr = (ctypes.c_double * len(norm))(*norm)
        h, ch = self.hdwf, ctypes.c_int(channel)
        node = ctypes.c_int(C.AnalogOutNodeCarrier)

        self.dwf.FDwfAnalogOutNodeEnableSet(h, ch, node, ctypes.c_int(1))
        self.dwf.FDwfAnalogOutNodeFunctionSet(h, ch, node, ctypes.c_ubyte(C.funcCustom))
        self.dwf.FDwfAnalogOutNodeDataSet(h, ch, node, arr, ctypes.c_int(len(norm)))
        self.dwf.FDwfAnalogOutNodeFrequencySet(h, ch, node, ctypes.c_double(frequency))
        self.dwf.FDwfAnalogOutNodeAmplitudeSet(h, ch, node, ctypes.c_double(peak))
        self.dwf.FDwfAnalogOutNodeOffsetSet(h, ch, node, ctypes.c_double(0.0))
        self.dwf.FDwfAnalogOutIdleSet(h, ch, ctypes.c_int(C.DwfAnalogOutIdleOffset))
        self.dwf.FDwfAnalogOutWaitSet(h, ch, ctypes.c_double(wait))
        self.dwf.FDwfAnalogOutRunSet(h, ch, ctypes.c_double(n_periods / frequency))
        self.dwf.FDwfAnalogOutRepeatSet(h, ch, ctypes.c_int(1))
        self.dwf.FDwfAnalogOutConfigure(h, ch, ctypes.c_int(1))

    def stop_wave(self, channel: int = C.CH_W1):
        self.dwf.FDwfAnalogOutConfigure(self.hdwf, ctypes.c_int(channel), ctypes.c_int(0))

    # ------------------------------------------------------------------ acquisition
    def arm_capture(self, sample_rate: float, n_samples: int,
                    trigger_on: int | None = C.CH_W1, ranges=(2.5, 2.5)):
        """
        Configure the scope for a single acquisition and arm it.

        trigger_on=0 means triggered by the start of W1 (Java's
        startAnalogCaptureBothChannelsTriggerOnWaveformGenerator),
        trigger_on=None means immediate free-running acquisition.
        """
        n_samples = min(int(n_samples), self.MAX_BUFFER)
        h = self.hdwf
        for ch, rng in zip((C.CH_SCOPE1, C.CH_SCOPE2), ranges):
            self.dwf.FDwfAnalogInChannelEnableSet(h, ctypes.c_int(ch), ctypes.c_int(1))
            self.dwf.FDwfAnalogInChannelRangeSet(h, ctypes.c_int(ch), ctypes.c_double(rng))

        self.dwf.FDwfAnalogInAcquisitionModeSet(h, ctypes.c_int(C.acqmodeSingle))
        self.dwf.FDwfAnalogInFrequencySet(h, ctypes.c_double(sample_rate))
        self.dwf.FDwfAnalogInBufferSizeSet(h, ctypes.c_int(n_samples))

        if trigger_on is None:
            self.dwf.FDwfAnalogInTriggerSourceSet(h, ctypes.c_ubyte(C.trigsrcNone))
        else:
            src = C.trigsrcAnalogOut1 if trigger_on == C.CH_W1 else C.trigsrcAnalogOut2
            self.dwf.FDwfAnalogInTriggerSourceSet(h, ctypes.c_ubyte(src))
            self.dwf.FDwfAnalogInTriggerAutoTimeoutSet(h, ctypes.c_double(0.0))
            # position 0 = trigger point in the middle of the buffer; here it is moved to the start of the buffer
            self.dwf.FDwfAnalogInTriggerPositionSet(
                h, ctypes.c_double(0.5 * n_samples / sample_rate)
            )

        self.dwf.FDwfAnalogInConfigure(h, ctypes.c_int(1), ctypes.c_int(1))
        self._n_samples = n_samples
        return n_samples

    def wait_until_armed(self, timeout: float = 5.0):
        t0 = time.time()
        sts = ctypes.c_byte()
        while True:
            self.dwf.FDwfAnalogInStatus(self.hdwf, ctypes.c_int(1), ctypes.byref(sts))
            if sts.value == C.DwfStateArmed:
                return
            if time.time() - t0 > timeout:
                raise DWFError("timed out waiting for the scope to arm")
            time.sleep(0.001)

    def wait_until_done(self, timeout: float = 10.0):
        t0 = time.time()
        sts = ctypes.c_byte()
        while True:
            self.dwf.FDwfAnalogInStatus(self.hdwf, ctypes.c_int(1), ctypes.byref(sts))
            if sts.value == C.DwfStateDone:
                return
            if time.time() - t0 > timeout:
                raise DWFError("acquisition timed out (no trigger? arm first, then start the waveform)")
            time.sleep(0.001)

    def record(self, sample_rate: float, n_samples: int,
               trigger_on: int | None = C.CH_W1, ranges=(2.5, 2.5),
               start_output=None, timeout: float = 60.0):
        """
        Long acquisition in Record mode -- for slow sweeps longer than one buffer (8192).

        The AD2's single-acquisition buffer has only 8192 points per channel. A slow DC sweep easily takes several hundred ms
        per ramp over several cycles, which exceeds it. Record mode streams samples back to the host,
        with length limited only by USB bandwidth and memory.

        start_output : optional callback, called after the acquisition is armed and before streaming starts,
                       to start the AWG (the wrong order loses the trigger).
        """
        h = self.hdwf
        n = int(n_samples)
        for ch, rng in zip((C.CH_SCOPE1, C.CH_SCOPE2), ranges):
            self.dwf.FDwfAnalogInChannelEnableSet(h, ctypes.c_int(ch), ctypes.c_int(1))
            self.dwf.FDwfAnalogInChannelRangeSet(h, ctypes.c_int(ch), ctypes.c_double(rng))

        length = n / sample_rate
        self.dwf.FDwfAnalogInAcquisitionModeSet(h, ctypes.c_int(C.acqmodeRecord))
        self.dwf.FDwfAnalogInFrequencySet(h, ctypes.c_double(sample_rate))
        self.dwf.FDwfAnalogInRecordLengthSet(h, ctypes.c_double(length))

        if trigger_on is None:
            self.dwf.FDwfAnalogInTriggerSourceSet(h, ctypes.c_ubyte(C.trigsrcNone))
        else:
            src = C.trigsrcAnalogOut1 if trigger_on == C.CH_W1 else C.trigsrcAnalogOut2
            self.dwf.FDwfAnalogInTriggerSourceSet(h, ctypes.c_ubyte(src))
            self.dwf.FDwfAnalogInTriggerAutoTimeoutSet(h, ctypes.c_double(0.0))
            self.dwf.FDwfAnalogInTriggerPositionSet(h, ctypes.c_double(0.5 * length))

        self.dwf.FDwfAnalogInConfigure(h, ctypes.c_int(1), ctypes.c_int(1))

        if start_output is not None:
            self.wait_until_armed()
            start_output()

        buf1 = (ctypes.c_double * n)()
        buf2 = (ctypes.c_double * n)()
        sts = ctypes.c_byte()
        avail, lost, corrupt = ctypes.c_int(), ctypes.c_int(), ctypes.c_int()
        got, n_lost, n_corrupt = 0, 0, 0
        t0 = time.time()

        while got < n:
            if not self.dwf.FDwfAnalogInStatus(h, ctypes.c_int(1), ctypes.byref(sts)):
                self._err("FDwfAnalogInStatus failed")
            if got == 0 and sts.value in (C.DwfStateConfig, C.DwfStatePrefill, C.DwfStateArmed):
                if time.time() - t0 > timeout:
                    raise DWFError("Record timed out waiting for trigger")
                continue

            self.dwf.FDwfAnalogInStatusRecord(
                h, ctypes.byref(avail), ctypes.byref(lost), ctypes.byref(corrupt)
            )
            got += lost.value
            n_lost += lost.value
            n_corrupt += corrupt.value
            if avail.value == 0:
                if time.time() - t0 > timeout:
                    raise DWFError("Record acquisition timed out")
                continue
            take = min(avail.value, n - got)
            off = ctypes.sizeof(ctypes.c_double) * got
            self.dwf.FDwfAnalogInStatusData(
                h, ctypes.c_int(C.CH_SCOPE1), ctypes.byref(buf1, off), ctypes.c_int(take))
            self.dwf.FDwfAnalogInStatusData(
                h, ctypes.c_int(C.CH_SCOPE2), ctypes.byref(buf2, off), ctypes.c_int(take))
            got += take

        if n_lost or n_corrupt:
            print(f"Warning: Record lost {n_lost} and corrupted {n_corrupt} samples -- "
                  f"sample rate {sample_rate/1e3:.1f} kHz is too high for USB; lower it and retry")

        self._n_samples = n
        return list(buf1), list(buf2)

    def fetch(self, n: int | None = None):
        """Fetch the raw data of both channels (no polarity inversion), returning (v1_raw, v2_raw)."""
        n = n or self._n_samples
        b1 = (ctypes.c_double * n)()
        b2 = (ctypes.c_double * n)()
        self.dwf.FDwfAnalogInStatusData(self.hdwf, ctypes.c_int(C.CH_SCOPE1), b1, ctypes.c_int(n))
        self.dwf.FDwfAnalogInStatusData(self.hdwf, ctypes.c_int(C.CH_SCOPE2), b2, ctypes.c_int(n))
        return list(b1), list(b2)

    # ------------------------------------------------------------------ conversion to physical quantities
    def to_physical(self, v1_raw, v2_raw):
        """
        Raw scope readings → (memristor voltage, current).

        V2:  V_m = -V(1+)              I = (V(1+) - V(2+)) / Rs
        V0/V1: V_m = V(1+) - V(2+)     I =  V(2+) / Rs
        """
        if self.board_version == 2:
            vm = [-a for a in v1_raw]
            cur = [(a - b) / self.rs for a, b in zip(v1_raw, v2_raw)]
        else:
            vm = [a - b for a, b in zip(v1_raw, v2_raw)]
            cur = [b / self.rs for b in v2_raw]
        return vm, cur

    # ------------------------------------------------------------------ high-level experiments
    def sweep(self, amplitude=1.0, frequency=100.0, waveform="triangle",
              offset=0.0, periods=2, samples=8000):
        """
        Hysteresis loop sweep: drive one period of a waveform and acquire the complete waveform synchronously.

        ★ **This is the path on which the previous chip died.** It drew about 455 µA at the time.

        A sweep differs fundamentally from a pulse: after a pulse the gate can read again and decide the next one,
        whereas during a sweep the resistance keeps changing and the gate cannot intervene. So authorisation here uses
        **the most pessimistic assumption**: the device is assumed to drop to the floor during the sweep, and the amplitude is computed for that resistance.

        In fact the current-limiting layer already covers the whole sweep automatically -- current = V/(Rs+R) is maximal at R→0,
        equal to I_safe, so no sweep, however long, can cross the line. Pessimistic authorisation makes the
        **device-voltage** layer hold throughout as well.

        Returns (t, v_memristor, current)  -- units s / V / A
        """
        amplitude = self._authorize_sweep(amplitude, "sweep", frequency, periods)
        sample_rate = frequency * samples / periods
        self.arm_capture(sample_rate, samples, trigger_on=C.CH_W1)
        self.wait_until_armed()
        self.start_wave(waveform, frequency, amplitude, offset)
        self.wait_until_done(timeout=10 + periods / frequency)
        v1, v2 = self.fetch()
        self.stop_wave()
        vm, cur = self.to_physical(v1, v2)
        t = [i / sample_rate for i in range(len(vm))]
        return t, vm, cur

    def sweep_dc(self, amplitude=0.5, period_ms=5.0, ramps=2,
                 shape="triangleupdown", samples_per_ramp=400,
                 waveform_points=4096):
        """
        Slow DC sweep (the GUI's DC experiment).

        The amplitude also passes the gate with the most pessimistic resistance (see the notes of `sweep()`).

        Difference from `sweep()`: `sweep()` uses the AWG's built-in standard waveforms for fast AC hysteresis;
        here a custom ramp with a ms-scale period is used for a slow sweep, showing quasi-static
        set/reset threshold behaviour rather than frequency-dependent hysteresis loops.

        Parameters
        ----------
        amplitude       : peak voltage (V), memristor coordinates. The DC experiment starts at 0.1 V by default.
        period_ms       : one ramp period in ms. The GUI range is 10–1000 ms,
                          default 5 ms. Period → frequency f = 1000/period_ms.
        ramps           : number of periods.
        shape           : sawtooth / sawtoothupdown / triangle / triangleupdown
        samples_per_ramp: samples per ramp. Hard-coded as 200 in Java;
                          when the total exceeds 8192 this method switches to Record mode automatically.

        Returns (t, v_memristor, current), units s / V / A
        """
        amplitude = self._authorize_sweep(amplitude, "sweep_dc",
                                          1000.0 / float(period_ms), ramps)
        freq = 1000.0 / float(period_ms)
        n_total = int(samples_per_ramp * ramps)
        fs = freq * samples_per_ramp
        wave = make_ramp(shape, amplitude, waveform_points)

        def _start():
            self.start_custom(wave, freq, n_periods=ramps)

        if n_total <= self.MAX_BUFFER:
            self.arm_capture(fs, n_total, trigger_on=C.CH_W1)
            self.wait_until_armed()
            _start()
            self.wait_until_done(timeout=10 + ramps / freq)
            v1, v2 = self.fetch()
        else:
            # beyond one buffer: use streaming Record
            v1, v2 = self.record(fs, n_total, trigger_on=C.CH_W1,
                                 start_output=_start,
                                 timeout=30 + 2 * ramps / freq)

        self.stop_wave()
        vm, cur = self.to_physical(v1, v2)
        t = [i / fs for i in range(len(vm))]
        return t, vm, cur

    # ------------------------------------------------------------------ device protection
    def attach_gate(self, gate):
        """
        Attach the protection gate. **Without it no voltage can be applied** -- every write/read path refuses to work.

        Normally do not call this by hand: use `hw_session.open_board()`,
        which handles the Rs check, preflight, gate and power-down together.
        """
        if gate.env.rs != self.rs:
            raise RuntimeError(
                f"The gate's Rs ({gate.env.rs/1e3:.1f} kΩ) and the driver's Rs "
                f"({self.rs/1e3:.1f} kΩ) differ. Current limiting rests entirely on this number; "
                f"two versions are not allowed.")
        self.gate = gate
        return self

    def _require_gate(self, what):
        if self.gate is None:
            raise RuntimeError(
                f"{what}: no protection gate attached; refusing to apply voltage.\n"
                f"  Get the board with hw_session.open_board(rs=..., chip_id=...); "
                f"it sets up the envelope, runs preflight and attaches the gate.\n"
                f"  This check cannot be bypassed -- the previous chip was pushed into the irreversible region "
                f"without a gate.")
        return self.gate

    def _closed_devices(self):
        """Which devices (0-based) the current DIO mask connects. The gate uses this to know what is being written."""
        n = 16 if self.board_version == 2 else 8
        return [i for i in range(n) if (self._dio_state >> i) & 1]

    def _authorize_sweep(self, amplitude, what, frequency, periods):
        gate = self._require_gate(what)
        dur = periods / max(frequency, 1e-9)
        return gate.authorize_sweep(self._closed_devices(), amplitude, dur)

    def probe_divider(self, levels=(0.2, 0.4, 0.6, 0.8, 1.0), n_avg=6,
                      width_us=50.0, verbose=False, range_v=2.5):
        """
        Raw data of the no-load divider measurement. All switches open; (source, node) measured at each voltage step.

        ★ **All steps must use the same fixed range.** The first version let the range follow the amplitude
        (rng = |v|×2.5), but the AD2 channel offset **changes with range** --
        which broke the premise that "the offset is constant and falls into the intercept of the fit".
        Measured residuals show a step exactly at the two range-switching steps (0.8 / 1.0 V).

        Returns [(v_src, v_node), ...]. Shared by `verify_rs` and `check_rs.py`.
        """
        import dwfconst as C
        gate = self._require_gate("probe_divider")
        self.all_off()
        time.sleep(0.01)
        out = []
        for v_set in levels:
            acc_n = acc_s = 0.0
            for _ in range(n_avg):
                v = gate.authorize_probe(self._dio_state, v_set)
                wave = make_pulse("square", v, duty=0.5, n=300)
                freq = 0.5 / (width_us * 1e-6)
                rng = float(range_v)        # fixed range, see the notes above
                self.arm_capture(freq * 300, 300, trigger_on=C.CH_W1,
                                 ranges=(rng, rng))
                self.wait_until_armed()
                self.start_custom(wave, freq, n_periods=1)
                self.wait_until_done(timeout=2.0)
                r1, r2 = self.fetch()
                self.stop_wave()
                seg = slice(60, 140)        # flat part of the pulse, avoiding the RC rising edge
                acc_n += sum(r1[seg]) / len(r1[seg])
                acc_s += sum(r2[seg]) / len(r2[seg])
            vs, vn = acc_s / n_avg, acc_n / n_avg
            out.append((vs, vn))
            if verbose:
                print(f"    set {v_set:4.2f} V → source {vs*1e3:+8.2f} mV  "
                      f"node {vn*1e3:+8.2f} mV  difference {(vs-vn)*1e3:+7.3f} mV")
        self.all_off()
        return out

    def measure_tau(self, v_probe=1.0, width_us=100.0, sample_rate=50e6,
                    n_samples=4000, n_rep=5, verbose=False):
        """
        **Channel-difference transient** of a no-load step -- the measurement that can actually determine Rs.

        =================== Only the right observable carries signal ===================

        The first two versions looked in the wrong place. The first looked at the 0.1 V DC difference (1.96 mV, buried in offset);
        the second looked at each channel's own rise time (smeared by point-by-point averaging of trigger jitter, so both channels
        gave the same number and the node was even "faster than the source" -- physically impossible, the fingerprint of the smearing).

        The right observable is **the difference of the two channels**:

            d(t) = V_src(t) − V_node(t)  =  I(t) · Rs

        At the instant of the step the capacitor is not yet charged; the node is at 0 while the source is already at V ⇒ **all of V is across Rs**.
        So d on the edge is **of order volts**, three orders of magnitude stronger than the DC plateau difference:

            Rs ≈ 0     →  node follows source  →  d ≈ 0 throughout
            Rs = 10 kΩ →  d peak ≈ V, τ = 1.8 µs
            Rs = 20 kΩ →  d peak ≈ V, τ = 3.6 µs
            Rs = 50 kΩ →  d peak ≈ V, τ = 9.0 µs

        This gives two independent criteria:

            **peak of d** -- whether Rs exists (≈0 is the dangerous branch)
            **τ = ∫d dt / d_peak** -- how large Rs is (area over peak, equivalent to
            ∫exp(−t/τ)dt = τ; **insensitive to edge shape and trigger jitter**)

            Rs = τ / C_par

        =================== Two pitfalls in the implementation ===================

        · **No point-by-point averaging.** Both channels are acquired by the same trigger at the same time, so jitter is
          common mode in d(t) and cancels; but averaging point by point across acquisitions widens the edge.
          So τ is computed for each acquisition and the **median** is taken.
        · **The waveform keeps a leading stretch of zeros**, so the edge falls in the middle of the buffer rather than at its edge,
          and both baseline and plateau can be measured.

        No load, no device in the circuit -- zero risk to the chip.
        """
        import dwfconst as C
        gate = self._require_gate("measure_tau")
        self.all_off()
        time.sleep(0.01)

        N_WAVE = 300
        LEAD = 120                      # number of leading zeros: the edge falls at 40% of the period
        dt = 1.0 / sample_rate
        taus, peaks, areas, drops = [], [], [], []

        for _ in range(max(1, n_rep)):
            v = gate.authorize_probe(self._dio_state, v_probe)
            wave = [0.0] * LEAD + [v] * (N_WAVE - LEAD)
            freq = 1.0 / (width_us * 1e-6)      # one full period = width_us
            rng = max(0.5, min(5.0, abs(v) * 2.5))
            n = self.arm_capture(sample_rate, n_samples,
                                 trigger_on=C.CH_W1, ranges=(rng, rng))
            self.wait_until_armed()
            self.start_custom(wave, freq, n_periods=1)
            self.wait_until_done(timeout=2.0)
            r1, r2 = self.fetch(n)          # r1 = node, r2 = source
            self.stop_wave()

            S = [abs(x) for x in r2]
            Nd = [abs(x) for x in r1]
            d = [a - b for a, b in zip(S, Nd)]
            k = len(d)
            if k < 50:
                continue

            i_pk = max(range(k), key=lambda i: d[i])
            # plateau: a stretch far enough after the edge (skipping 10× the maximum expected τ)
            skip = min(k - 1, i_pk + int(60e-6 / dt))
            tail = d[skip:] or d[-10:]
            d_pl = sorted(tail)[len(tail) // 2]
            peak = d[i_pk] - d_pl
            if peak <= 0:
                continue

            # area: integrate from the peak until it falls back to 2% of the peak
            area = 0.0
            for i in range(i_pk, k):
                x = d[i] - d_pl
                if x <= 0.02 * peak:
                    break
                area += x * dt
            if area <= 0:
                continue

            dv = (sorted(S[skip:])[len(S[skip:]) // 2]
                  if k - skip > 2 else max(S)) - (
                  sorted(S[:max(3, LEAD // 3)])[0])
            taus.append(area / peak)
            peaks.append(peak)
            areas.append(area)
            drops.append(d_pl)
            if verbose:
                print(f"    one acquisition: d peak {peak*1e3:8.1f} mV  "
                      f"area {area*1e6:7.3f} µV·s  τ = {area/peak*1e6:6.3f} µs  "
                      f" (step amplitude {dv*1e3:.0f} mV, peak/step = "
                      f"{peak/max(dv,1e-9):.3f})")
        self.all_off()

        out = {"dt": dt, "cpar": self.cpar, "n_rep": len(taus)}
        if not taus:
            out["status"] = "no_edge"
            if verbose:
                print("    No step edge captured. Check the trigger and sample rate.")
            return out

        med = lambda a: sorted(a)[len(a) // 2]
        tau, peak, area = med(taus), med(peaks), med(areas)
        out.update({"tau": tau, "d_peak": peak, "area": area,
                    "d_plateau": med(drops),
                    "rs_implied": tau / self.cpar if self.cpar > 0 else float("inf"),
                    "tau_spread": (max(taus) - min(taus)) / max(tau, 1e-12),
                    "peak_frac": peak / max(abs(v_probe), 1e-9)})

        # resolution: τ must be clearly larger than one sample interval, and the peak must clearly exist
        out["resolved"] = (tau > 5 * dt and out["peak_frac"] > 0.05
                           and out["tau_spread"] < 0.6)
        out["status"] = "ok" if out["resolved"] else "unresolved"

        if verbose:
            print(f"    median: d peak {peak*1e3:.1f} mV "
                  f" ({out['peak_frac']*100:.1f}% of the step), "
                  f"area {area*1e6:.3f} µV·s")
            print(f"    ⇒ τ = {tau*1e6:.3f} µs ({len(taus)} acquisitions, spread "
                  f"{out['tau_spread']*100:.0f}%, {tau/dt:.0f} sample points)")
            print(f"    ⇒ inverted with C_par = {self.cpar*1e12:.0f} pF "
                  f"Rs ≈ {out['rs_implied']/1e3:.2f} kΩ")
        return out

    def verify_rs(self, tol=0.25, verbose=True, levels=None):
        """
        **Measure the true value of the series resistance directly.** All current limiting rests on this number.
        This is layer 5 of safety.py: it tests the only assumption layer 1 depends on.

        Principle: with all 16 switches open, the branch reduces to

            W1 ──[ Rs ]── node ──[ R_scope ≈ 1 MΩ scope input ]── GND

        so V_node / V_src = R_scope/(R_scope + Rs), which gives Rs.
        **There is no device in the circuit, zero risk to the chip**, so this measurement can use
        much higher voltage than a read pulse.

        ★ The first version of this method was wrong, and the reason is worth recording here:

          It used a single 0.10 V point and simply subtracted the means of the two channels. But the division of 20 kΩ against
          1 MΩ is only 2% -- **1.96 mV** at 0.1 V, while the AD2 channel offsets are tens of mV
          (which is what `ktram.calibrate_offsets` measures).
          That measurement measured the offset, not Rs. That is how the reported 2.3 kΩ arose:
          the difference "measured" only 0.23 mV, entirely in the floor.

          Two things are now changed:
            1. **Multi-point fit**. Measure (V_src, V_node) at several voltage steps and find the slope k by least squares
               through the origin. The offset is a constant and falls into the **intercept**, so it is removed automatically.
            2. **Raise the voltage to 1 V**. The difference goes from 1.96 mV to 19.6 mV, SNR ×10.
               At no load the current at 1 V is 1 µA, and it does not flow through the chip.

        Returns a dict whose status is "ok" / "mismatch" / "inconclusive".
        **"Cannot be measured" and "measured wrong" are different things** and must not both be treated as failure:
        the former means the instrument is not sensitive enough, the latter that board and code disagree.
        """
        levels = levels or (0.2, 0.4, 0.6, 0.8, 1.0)
        pts = self.probe_divider(levels, verbose=verbose)

        # --- least squares through the origin: k = Σ(xy)/Σ(x²), x=V_src, y=V_node
        sxx = sum(x * x for x, _ in pts)
        sxy = sum(x * y for x, y in pts)
        k = sxy / sxx if sxx > 0 else 0.0
        # --- fit with intercept, only to report the removed offset
        n = len(pts)
        mx = sum(x for x, _ in pts) / n
        my = sum(y for _, y in pts) / n
        den = sum((x - mx) ** 2 for x, _ in pts)
        k2 = (sum((x - mx) * (y - my) for x, y in pts) / den) if den > 0 else k
        b = my - k2 * mx
        resid = max(abs(y - (k2 * x + b)) for x, y in pts)

        # ★ linearity must look at the **difference** against V_src, not V_node against V_src.
        #   the latter spans 0.1~1.0 V on both sides, so R² is necessarily ≈ 1.000000 and says nothing about
        #   the tiny difference -- the first version reported exactly this useless R².
        #   a real divider must be a straight line through the origin; the two measured R² (difference vs V) were
        #   −4.29 and −0.81 (negative = worse than "difference equals its mean"),
        #   i.e. **the difference does not follow the voltage at all**.
        dps = [(x, x - y) for x, y in pts]
        s_xx = sum(x * x for x, _ in dps)
        s_xy = sum(x * dd for x, dd in dps)
        k_dp = s_xy / s_xx if s_xx > 0 else 0.0
        dbar = sum(dd for _, dd in dps) / len(dps)
        ss_tot = sum((dd - dbar) ** 2 for _, dd in dps)
        ss_res = sum((dd - k_dp * x) ** 2 for x, dd in dps)
        r2_drop = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

        rp = self.SCOPE_INPUT_R
        drop_frac = 1.0 - k2            # computed with the slope of the fit with intercept, which is the actual division
        rs_meas = rp * drop_frac / k2 if k2 > 0 else float("inf")
        # resolution: fit residual converted to an uncertainty in Rs
        v_max = max(abs(x) for x, _ in pts)
        rs_sigma = rp * (resid / max(v_max, 1e-9)) / max(k2, 1e-9)

        out = {"pts": pts, "k": k2, "offset": b, "resid": resid,
               "rs_meas": rs_meas, "rs_sigma": rs_sigma,
               "rs_declared": self.rs, "scope_r": rp,
               "k_drop": k_dp, "r2_drop": r2_drop}

        if verbose:
            print(f"  Rs self-check ({n}-point fit, offset removed as intercept)")
            print(f"    division ratio k = {k2:.5f}   intercept = {b*1e3:+.2f} mV"
                  f" (this is the offset the old single-point method measured)")
            print(f"    fit residual = {resid*1e3:.3f} mV")
            print(f"    linearity check: difference vs V_src, fit through origin, "
                  f"R² = {r2_drop:+.3f}"
                  + ("  ← only close to 1 shows a real division"
                     if r2_drop > 0.8 else
                     "  ← **difference does not follow the voltage ⇒ not a division, but offset/noise**"))

        # --- decision
        #
        # ★ the criterion here was also wrong in the first version, recorded: it compared the **measured** difference with the noise,
        #   and a small difference meant "no resolution". The right question is whether the **expected** difference is buried in noise.
        #   in that measurement: expected 1.96%, noise 0.0166% ⇒ SNR 118; if that division existed
        #   it would have been seen. So "no visible division" is a **definite conclusion**, not a failure to measure.
        noise_frac = resid / max(v_max, 1e-9)
        drop_expect = self.rs / (self.rs + rp)      # division expected under the 1 MΩ assumption
        out.update({"noise_frac": noise_frac, "drop_frac": drop_frac,
                    "drop_expect": drop_expect,
                    "snr_expect": drop_expect / max(noise_frac, 1e-12)})

        if verbose:
            print(f"    division: measured {drop_frac*100:.4f}%  /  "
                  f"expected at {rp/1e6:.0f} MΩ {drop_expect*100:.3f}%  /  "
                  f"noise floor {noise_frac*100:.4f}%")

        if drop_expect < 5 * noise_frac:
            # genuine "cannot measure": even the expected signal is buried in noise
            out["status"] = "inconclusive"
            if verbose:
                print(f"    → **Cannot be measured**: the expected division itself is buried in noise"
                      f" (SNR {out['snr_expect']:.1f}).")
            return out

        if drop_frac < drop_expect / 3:
            # the expected signal should have been visible but was not -- definite, but it **does not determine Rs**.
            # the DC division is degenerate in Rs: Rs≈0 and R_scope≫Rs give the same data.
            rp_implied = self.rs * k2 / max(drop_frac, 1e-12)
            out["status"] = "degenerate"
            out["rp_implied"] = rp_implied
            if verbose:
                print(f"    → **The division expected under the {rp/1e6:.0f} MΩ assumption is absent**"
                      f" (only 1/{drop_expect/max(drop_frac,1e-12):.0f} of expected).")
                print(f"      SNR {out['snr_expect']:.0f}; if that division existed it would have been seen, "
                      f"so this conclusion is definite -- ")
                print(f"      but it **cannot determine Rs**. The no-load division is degenerate in Rs, "
                      f"and the following two give the same data:")
                print(f"        (a) Rs ≈ {rs_meas/1e3:.2f} kΩ, i.e. **almost no current limiting**"
                      f"  ← dangerous branch")
                print(f"        (b) Rs is the declared {self.rs/1e3:.0f} kΩ, but the scope input impedance "
                      f"is {rp_implied/1e6:.0f} MΩ rather than {rp/1e6:.0f} MΩ")
                print(f"      To separate them, look at the **rise time**: τ = (Rs∥R_scope)·C_par "
                      f"≈ Rs·C_par,")
                print(f"      which depends only on Rs -- see measure_tau().")
            return out

        err = abs(rs_meas - self.rs) / self.rs
        out["err"] = err
        out["status"] = "ok" if err <= tol else "mismatch"
        if verbose:
            print(f"    → measured Rs ≈ {rs_meas/1e3:.1f} ± {rs_sigma/1e3:.1f} kΩ, "
                  f"the code says {self.rs/1e3:.1f} kΩ, deviation {err*100:.0f}%  "
                  f"{'✓' if out['status'] == 'ok' else '✗'}")
            if out["status"] == "mismatch":
                print(f"      ★ deviation above {tol*100:.0f}% -- the whole premise of current limiting is this number.")
                print(f"        Check the position of the Mode switch and that both series resistors at A/B are fitted,")
                print(f"        then measure that resistor with a multimeter to confirm.")
        return out

    def verify_rs_full(self, tol=0.25, verbose=True):
        """
        Run both methods together and cross-check. The DC division is degenerate in Rs and sensitive to R_scope;
        the rise time is sensitive to Rs and insensitive to R_scope. Only together do they determine it.

        Returns {"dc": ..., "rc": ..., "status": ..., "rs_best": ...}
        """
        if verbose:
            print(f"  Rs self-check -- method 1: DC division (no load)")
        dc = self.verify_rs(tol=tol, verbose=verbose)
        if verbose:
            print(f"  Rs self-check -- method 2: step rise time (no load, insensitive to R_scope)")
        rc = self.measure_tau(verbose=verbose)

        out = {"dc": dc, "rc": rc}
        # ★ the status of each side may be "weak" (degenerate / unresolved),
        #   but each still points to a value. **As long as the two values agree,
        #   the cross-check holds** -- they depend on two different calibration quantities,
        #   so agreement cannot be a coincidence.
        #   this bit us on real hardware: in Mode 2, DC gave 0.29 kΩ and RC 0.27 kΩ,
        #   7% apart, yet because both statuses were weak my code reported unverified,
        #   so --rs-dmm let through a board that "had no series resistor at all".
        rs_dc = dc.get("rs_meas")
        if rs_dc is not None and not (rs_dc > 0 and math.isfinite(rs_dc)):
            rs_dc = None
        rs_rc = rc.get("rs_implied")
        if rs_rc is not None and not (rs_rc > 0 and math.isfinite(rs_rc)):
            rs_rc = None

        # ★★ agreement of the two methods = **settled**.
        #
        #   they depend on two **different** calibration quantities: the DC division on R_scope (1 MΩ),
        #   the step transient on C_par (180 pF). Two independent constants giving the same Rs
        #   also cross-validate the two constants themselves. If either were wrong, the two methods would necessarily diverge.
        #
        #   measured: 9.3 kΩ (DC) vs 9.76 kΩ (transient), 5% apart -- consistent.
        # ★★ the parasitic-level check is done **first, independently of whether the two methods agree**.
        #
        #   at this level the relative spread of the two methods is naturally large (both are mV-level signals),
        #   so requiring 25% agreement is too strict. On real hardware two measurements gave
        #   (0.29, 0.27) and (0.17, 0.23) kΩ; the latter, 26% apart, was kept out of the cross-check,
        #   so only "large deviation" was reported rather than "no series resistor" -- far less informative.
        #
        #   right approach: compare the **looser (larger) estimate** with the parasitic level.
        #   if even the larger estimate is below parasitic level, the conclusion is definite, and in the conservative direction.
        r_par = self.TOTAL_PARASITIC_RESISTANCE + 2 * self.R_SWITCH
        cands = [x for x in (rs_dc, rs_rc) if x]
        if cands and max(cands) < 5 * r_par:
            out["rs_best"] = max(cands)
            out["r_parasitic"] = r_par
            out["cross_validated"] = (
                len(cands) == 2
                and abs(rs_dc - rs_rc) / max(rs_dc, rs_rc) < 0.5)
            out["err"] = abs(out["rs_best"] - self.rs) / self.rs
            out["status"] = "no_series_resistor"
            if verbose:
                est = " / ".join(f"{x/1e3:.2f}" for x in cands)
                print(f"  ★★★ measured Rs ≈ {est} kΩ (DC / transient), "
                      f"while the purely parasitic path")
                print(f"       (switch {self.R_SWITCH:.0f} Ω×2 + traces "
                      f"{self.TOTAL_PARASITIC_RESISTANCE:.0f} Ω) alone is "
                      f"{r_par:.0f} Ω.")
                print(f"       **There is no series resistor on this branch** -- no current limiting at all.")
            return out

        if rs_dc and rs_rc and abs(rs_dc - rs_rc) / max(rs_dc, rs_rc) < 0.25:
            rs_best = 0.5 * (rs_dc + rs_rc)
            out["rs_best"] = rs_best
            out["cross_validated"] = True
            out["err"] = abs(rs_best - self.rs) / self.rs
            out["status"] = "determined" if out["err"] <= tol else "mismatch"
            if verbose:
                print(f"  Cross-check of two methods: DC {rs_dc/1e3:.2f} kΩ (via R_scope)"
                      f" vs transient {rs_rc/1e3:.2f} kΩ (via C_par), "
                      f"{abs(rs_dc-rs_rc)/max(rs_dc,rs_rc)*100:.0f}% apart")
                print(f"    Two **independent** calibration quantities give the same value ⇒ Rs is settled, "
                      f"and the two constants themselves are cross-validated.")
        elif rs_dc:
            out["rs_best"], out["cross_validated"] = rs_dc, False
            out["err"] = abs(rs_dc - self.rs) / self.rs
            out["status"] = "ok" if out["err"] <= tol else "mismatch"
        elif rs_rc:
            out["rs_best"], out["cross_validated"] = rs_rc, False
            out["err"] = abs(rs_rc - self.rs) / self.rs
            out["status"] = "ok_rc" if out["err"] <= max(tol, 0.6) else "mismatch_rc"
        else:
            out["status"] = "unverified"
        if verbose:
            print(f"  → Overall verdict: {out['status']}"
                  + (f", Rs ≈ {out['rs_best']/1e3:.2f} kΩ"
                     if out.get("rs_best") else ""))
        return out

    def pulse(self, amplitude=1.0, width_us=50.0, n=1, shape="square",
              duty=0.5, samples_per_pulse=300, capture=True,
              forward=None, devices=None, _read=False):
        """
        Apply n pulses and (optionally) capture the waveform of the first pulse.

        The amplitude first passes the protection gate (safety.SafetyGate). The gate does more than the old clipping:
        besides limiting current and voltage, it checks **where the connected devices are now**, refuses in the SET direction
        to push below the floor, and requires fresh readings. See the five layers described in safety.py.

        forward : None means judged by the sign of amplitude (positive = SET).
        devices : None means those connected in the current DIO mask.
        _read   : internal; read pulses go through the tighter authorize_read path.

        Returns (t, v_memristor, current); returns None when capture=False.
        """
        gate = self._require_gate("pulse")
        devs = self._closed_devices() if devices is None else list(devices)
        if _read:
            amplitude = gate.authorize_read(devs, amplitude)
        else:
            fwd = (amplitude >= 0) if forward is None else bool(forward)
            amplitude = gate.authorize_write(
                devs, amplitude, width_us=width_us, forward=fwd, n=n)
        wave = make_pulse(shape, amplitude, duty, samples_per_pulse)
        # one waveform period = pulse width / duty. duty=0.5, width 50 µs → period 100 µs → 10 kHz.
        freq = duty / (width_us * 1e-6)

        if capture:
            sr = freq * samples_per_pulse
            self.arm_capture(sr, samples_per_pulse, trigger_on=C.CH_W1)
            self.wait_until_armed()

        self.start_custom(wave, freq, n_periods=n)

        if not capture:
            time.sleep(n / freq + 0.002)
            self.stop_wave()
            return None

        self.wait_until_done(timeout=5 + n / freq)
        v1, v2 = self.fetch()
        self.stop_wave()
        vm, cur = self.to_physical(v1, v2)
        t = [i / (freq * samples_per_pulse) for i in range(len(vm))]
        return t, vm, cur

    def read_resistance(self, index: int | None = None, v_read=0.1,
                        width_us=25.0, use_rc_model=True) -> float:
        """
        Measure the memristor resistance with a low-voltage read pulse, in Ω.

        With use_rc_model=True the RC model is inverted (Java's JSpice + lookup table),
        which is more accurate for high-resistance devices/short pulses; False uses a simple ohmic divider (fast, enough at low resistance).

        The reading is **fed back to the gate automatically** (note_resistance). This matters:
        the gate's floor rule requires a fresh reading before SET, and "fresh" can only come from here.
        So "read once before writing" is not a style suggestion but a hard precondition.
        """
        gate = self._require_gate("read_resistance")
        if index is not None:
            self.select(index)
        devs = self._closed_devices()

        t, vm, cur = self.pulse(amplitude=min(abs(v_read), gate.env.v_read),
                                width_us=width_us, n=1, _read=True)

        # take the flat part of the pulse (last 1/4 of the first half period, avoiding the rising edge) -- Java averages the first half
        half = len(vm) // 2
        seg = slice(int(half * 0.5), half)
        v_node = sum(vm[seg]) / max(1, len(vm[seg]))

        if not use_rc_model:
            i_avg = sum(cur[seg]) / max(1, len(cur[seg]))
            r = (float("inf") if abs(i_avg) < 1e-12 else
                 abs(v_node / i_avg) - self.TOTAL_PARASITIC_RESISTANCE)
        else:
            from rc_resistance import RCResistanceModel
            model = RCResistanceModel(
                series_resistance=self.rs,
                pulse_amplitude=min(abs(v_read), gate.env.v_read),
                pulse_width=width_us * 1e-6,
                capacitance=self.cpar,
                scope_resistance=self.SCOPE_INPUT_R,
            )
            r = model.resistance_from_voltage(abs(v_node))

        # feed back: single device via note_resistance, several devices via note_branch (the parallel value is a lower bound)
        if len(devs) == 1:
            gate.note_resistance(devs[0], r)
        elif devs:
            gate.note_branch(devs, r)
        return r

    def read_branch_resistance(self, index, v_read=0.1, width_us=25.0,
                               offsets=(0.0, 0.0, 0.0), n_avg=8):
        """
        **Read a single device in Mode 2.** Devices 0..7 are on branch A (scope 1+),
        8..15 on branch B (scope 2+).

        ★ Why this is needed: `read_resistance()` looks only at scope 1+; it was written for
          **the single-node topology of Mode 1** -- all 16 devices on one node.
          In Mode 2, 8..15 hang on branch B and scope 1+ cannot see them at all,
          so `read_all()` reports them all as "open". On real hardware 8 were misreported this way.

        In Mode 2 both channels measure nodes (neither measures the source), so the source voltage can only be
        **assumed**: w1 = polarity·v_read + offw. This is the same formula as ktram.HardwareBackend.read(),
        so the two paths are consistent.

            R = Rs · p / (w1 − p)

        offsets = (scope1_offset, scope2_offset, w1_offset),
        measured by ktram.calibrate_offsets(). Without compensation, 5 mV of offset on an 80 mV read signal
        causes resistance errors of tens of percent.
        """
        gate = self._require_gate("read_branch_resistance")
        import dwfconst as C
        i = int(index)
        self.select(i)
        off1, off2, offw = offsets
        v = min(abs(v_read), gate.env.v_read)
        v = gate.authorize_read([i], v)

        # average several reads: random noise falls as √n, raising the resolution limit.
        # the 0.10 V read pulse is 38% of threshold and does not disturb the device, so more reads cost nothing.
        acc1 = [0.0] * 300
        acc2 = [0.0] * 300
        n_avg = max(1, int(n_avg))
        for k in range(n_avg):
            if k:
                v = gate.authorize_read([i], v)
            wave = make_pulse("square", v, duty=0.5, n=300)
            freq = 0.5 / (width_us * 1e-6)
            self.arm_capture(freq * 300, 300, trigger_on=C.CH_W1,
                             ranges=(0.5, 0.5))
            self.wait_until_armed()
            self.start_custom(wave, freq, n_periods=1)
            self.wait_until_done(timeout=2.0)
            a, b = self.fetch()
            for t in range(min(300, len(a))):
                acc1[t] += a[t]
                acc2[t] += b[t]
            self.stop_wave()
        r1 = [x / n_avg for x in acc1]
        r2 = [x / n_avg for x in acc2]

        seg = slice(60, 140)
        raw = (r1 if i < 8 else r2)
        p = sum(raw[seg]) / len(raw[seg]) - (off1 if i < 8 else off2) - offw
        w1 = self._polarity() * v + offw
        d = w1 - p

        # ★ when the drop is below the noise threshold, the conclusion is not "open" but a **lower bound**:
        #
        #     drop = |w1|·Rs/(R+Rs) < thr  ⇒  R > Rs·(|w1|/thr − 1)
        #
        #   0.10 V read level, Rs = 20 kΩ, thr = 0.4 mV ⇒ R > 5.0 MΩ,
        #   250× above the 20 kΩ floor -- fully acceptable for safety.
        #
        #   I had already used this in ktram._branch_r, but missed it on this read path:
        #   on real hardware device 15 was therefore judged "open/unusable", although it was only **high resistance beyond
        #   read resolution**. Judging high resistance as a bad device is wrong, and it blocks preflight.
        #
        #   note that this resolution limit is not high (see the table below), so the high end of readings is always
        #   treated only as a lower bound:
        #     R = 1 MΩ → drop 1.96 mV      R = 2 MΩ → 0.99 mV
        #     R = 5 MΩ → 0.40 mV  ← same order as the offset residual
        thr = self.READ_NOISE_V
        r_detect = self.rs * (abs(w1) / thr - 1.0) if thr > 0 else float("inf")
        bounded = False
        if abs(d) < thr or (p / w1 if w1 else 0) <= 0:
            r, bounded = r_detect, True
        else:
            r = self.rs * p / d
            if r <= 0:
                r, bounded = r_detect, True
        if bounded:
            self.read_bounded.add(i)
        else:
            self.read_bounded.discard(i)
        gate.note_resistance(i, r)
        return r

    def read_all_mode2(self, v_read=0.1, width_us=25.0,
                       offsets=(0.0, 0.0, 0.0), n_avg=8):
        """Read all 16 devices in Mode 2 (side A via scope 1+, side B via scope 2+)."""
        self.read_bounded = set()
        out = [self.read_branch_resistance(i, v_read, width_us, offsets, n_avg)
               for i in range(16)]
        self.all_off()
        return out

    def read_all(self, v_read=0.1, width_us=25.0, use_rc_model=True):
        """
        Go through all devices on the chip and return a list of resistances (Ω).

        This is also the standard action for refreshing the gate's freshness: one read_all
        feeds back readings for all 16 devices, and only then is SET allowed.
        """
        n = 16 if self.board_version == 2 else 8
        out = []
        for i in range(n):
            self.select(i)
            out.append(self.read_resistance(v_read=v_read, width_us=width_us,
                                            use_rc_model=use_rc_model))
        self.all_off()
        return out


# ---------------------------------------------------------------------------
# waveform generation (WaveformUtils.java)
# ---------------------------------------------------------------------------
def make_pulse(shape: str, amplitude: float, duty: float = 0.5, n: int = 300):
    """
    Generate one period of a pulse waveform (voltage sequence).
    The first duty fraction is the pulse, followed by zeros -- so the AWG returns to 0 V between pulses.
    """
    shape = shape.lower()
    on = max(1, int(round(n * duty)))
    out = [0.0] * n
    for k in range(on):
        x = k / on
        if shape == "square":
            y = 1.0
        elif shape == "triangle":
            y = 1 - abs(2 * x - 1)
        elif shape in ("halfsine", "half_sine"):
            y = math.sin(math.pi * x)
        elif shape in ("quartersine", "quarter_sine"):
            y = math.sin(math.pi / 2 * x)
        elif shape in ("squaredecay", "square_decay"):
            y = 1.0 if x < 0.5 else math.exp(-(x - 0.5) * 10)
        elif shape in ("squaresmooth", "square_smooth"):
            edge = 0.1
            y = min(1.0, x / edge, (1 - x) / edge)
        else:
            raise ValueError(f"unknown waveform {shape}")
        out[k] = amplitude * y
    return out


def make_ramp(shape: str, amplitude: float, n: int = 4096, offset: float = 0.0):
    """
    Ramp waveform for the DC experiment, one full period.

    One-to-one with Java's core/driver/waveform/*.java (x = t/T ∈ [0,1)):

      sawtooth        0 → +A            monotonic rise, jumps back to 0 at the end of the period
      sawtoothupdown  0 → +A ⇢ 0 → −A   positive rise, jump back to 0, then negative rise
      triangle        0 → +A → 0        unipolar triangle
      triangleupdown  0 → +A → −A → 0   bipolar triangle (the GUI defaults to SawtoothUpDown,
                                        but this is more convenient for set/reset thresholds)
    """
    shape = shape.lower().replace("_", "")
    A = float(amplitude)
    out = []
    for k in range(n):
        x = k / n
        if shape == "sawtooth":
            y = A * x
        elif shape == "sawtoothupdown":
            y = 2 * A * x if x < 0.5 else -2 * A * x + A
        elif shape == "triangle":
            y = 2 * A * x if x < 0.5 else -2 * A * x + 2 * A
        elif shape == "triangleupdown":
            if x < 0.25:
                y = 4 * A * x
            elif x >= 0.75:
                y = 4 * A * x - 4 * A
            else:
                y = -4 * A * x + 2 * A
        else:
            raise ValueError(f"unknown ramp waveform {shape}; "
                             f"options are sawtooth / sawtoothupdown / triangle / triangleupdown")
        out.append(offset + y)
    return out


def make_sweep(shape: str, amplitude: float, n: int = 8000, offset: float = 0.0):
    """Generate one full period of a sweep waveform for a custom hysteresis sweep."""
    shape = shape.lower()
    out = []
    for k in range(n):
        x = k / n
        if shape == "sine":
            y = math.sin(2 * math.pi * x)
        elif shape == "triangle":
            # 0 -> +1 -> 0, positive half only
            y = 2 * x if x < 0.5 else 2 - 2 * x
        elif shape == "triangleupdown":
            # 0 -> +1 -> 0 -> -1 -> 0
            if x < 0.25:
                y = 4 * x
            elif x < 0.75:
                y = 2 - 4 * x
            else:
                y = 4 * x - 4
        elif shape == "sawtooth":
            y = 2 * x - 1
        else:
            raise ValueError(f"unknown waveform {shape}")
        out.append(offset + amplitude * y)
    return out
