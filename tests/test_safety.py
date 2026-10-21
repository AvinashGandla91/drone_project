from drone_safety.safety import SafetyGate

def test_direction_needs_two_frames():
    g = SafetyGate()
    assert g.process("LEFT", .9, .9, "HIGHER STABILITY").decision == "PENDING_CONFIRMATION"
    assert g.process("LEFT", .9, .9, "HIGHER STABILITY").accepted_command == "LEFT"

def test_conflicting_direction_resets_confirmation():
    g = SafetyGate()
    g.process("LEFT", .9, .9, "HIGHER STABILITY")
    assert g.process("RIGHT", .9, .9, "HIGHER STABILITY").decision == "PENDING_CONFIRMATION"
    assert g.active_command is None

def test_stop_immediate_and_cancels_pending():
    g = SafetyGate()
    g.process("LEFT", .9, .9, "HIGHER STABILITY")
    d = g.process("STOP_HOLD", .1, .1, "LOW MOVEMENT STABILITY")
    assert d.accepted_command == "STOP_HOLD"
    assert d.decision == "ACCEPTED_IMMEDIATE_OVERRIDE"
    assert g.process("LEFT", .9, .9, "HIGHER STABILITY").decision == "PENDING_CONFIRMATION"

def test_unknown_and_low_confidence_rejected():
    g = SafetyGate()
    assert g.process("UNKNOWN", .99, .9, "HIGHER STABILITY").accepted_command is None
    assert g.process("LEFT", .74, .9, "HIGHER STABILITY").accepted_command is None

def test_stability_gates_direction():
    assert SafetyGate().process("LEFT", .99, .59, "LOW MOVEMENT STABILITY").decision == "HELD_LOW_STABILITY"

def test_land_separate_and_gated():
    assert SafetyGate().process("LAND", .99, .9, "HIGHER STABILITY").accepted_command == "LAND"
    assert SafetyGate().process("LAND", .99, .65, "UNCERTAIN").accepted_command is None
