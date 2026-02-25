# IONIS V22-gamma + PhysicsOverrideLayer Test Suite
#
# Test Groups:
#   KI7MT Override: 18 operator-grounded tests (17 hard + 1 soft)
#     - 16 single-path tests (positive/negative physics gates)
#     - 2 band ordering tests (day/night)
#     - 4 validation gates: raw=16/17, override=17/17, 0 regressions, acid FAIL->PASS
#
#   TST-900: 11 band x time discrimination tests
#     - Band closure (10m, 15m winter day/night)
#     - Mutual darkness (160m, 80m night vs day)
#     - Band ordering (day: high > low, night: low > high)
#     - Time sensitivity, peak hours, gray line
#
# Total: 29 tests
#
# Run all tests:
#   ionis-validate test
#
