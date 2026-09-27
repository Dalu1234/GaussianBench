"""
Controller mapping wizard v2  -  settles before each test.
"""
import pygame, time

pygame.init()
pygame.joystick.init()

if pygame.joystick.get_count() == 0:
    print("No controller found."); exit()

joy = pygame.joystick.Joystick(0)
joy.init()
print(f"Found: {joy.get_name()}  ({joy.get_numaxes()} axes, {joy.get_numbuttons()} buttons)\n")

def settle(secs=1.5):
    """Drain events and wait for axes to stop drifting."""
    t0 = time.time()
    while time.time() - t0 < secs:
        pygame.event.pump()
        time.sleep(0.05)
    return [joy.get_axis(i) for i in range(joy.get_numaxes())]

def detect_axis(prompt, timeout=6):
    print(f"  >>> {prompt}")
    print(f"      (settling 1.5s  -  keep everything still...)", end="\r")
    baseline = settle(1.5)
    print(f"      Ready! You have {timeout}s.                  ")
    best_axis, best_delta, best_raw = None, 0.0, 0.0
    t0 = time.time()
    while time.time() - t0 < timeout:
        pygame.event.pump()
        for i in range(joy.get_numaxes()):
            v = joy.get_axis(i)
            d = abs(v - baseline[i])
            if d > best_delta:
                best_delta, best_axis, best_raw = d, i, v
        time.sleep(0.03)
    if best_axis is not None and best_delta > 0.3:
        print(f"      -> Axis {best_axis}  (delta={best_delta:.2f}, raw={best_raw:.2f})\n")
        return best_axis, best_raw
    print(f"      -> Nothing detected\n")
    return None, None

def detect_button(prompt, timeout=6):
    print(f"  >>> {prompt}")
    settle(0.5)
    print(f"      Ready! You have {timeout}s.")
    t0 = time.time()
    while time.time() - t0 < timeout:
        pygame.event.pump()
        for i in range(joy.get_numbuttons()):
            if joy.get_button(i):
                print(f"      -> Button {i}\n")
                time.sleep(0.4)
                return i
        time.sleep(0.03)
    print(f"      -> Nothing detected\n")
    return None

print("=" * 52)
print("  CONTROLLER MAPPING WIZARD  (keep still between steps)")
print("=" * 52 + "\n")

lsx_axis, lsx_raw  = detect_axis("Push LEFT STICK fully RIGHT then back to centre")
lsz_axis, lsz_raw  = detect_axis("Push LEFT STICK fully DOWN then back to centre")
rt_axis,  rt_raw   = detect_axis("Pull RIGHT TRIGGER fully then release")
lt_axis,  lt_raw   = detect_axis("Pull LEFT TRIGGER fully then release")

btn_a  = detect_button("Press A")
btn_b  = detect_button("Press B")
btn_x  = detect_button("Press X")
btn_y  = detect_button("Press Y")
btn_lb = detect_button("Press LB")
btn_rb = detect_button("Press RB")

print("=" * 52)
print("  FINAL MAPPING")
print("=" * 52)
print(f"  Left stick X    axis {lsx_axis}   (right = {lsx_raw:.2f})")
print(f"  Left stick Z    axis {lsz_axis}   (down  = {lsz_raw:.2f})")
print(f"  Right trigger   axis {rt_axis}   (full  = {rt_raw:.2f})")
print(f"  Left trigger    axis {lt_axis}   (full  = {lt_raw:.2f})")
print(f"  A={btn_a}  B={btn_b}  X={btn_x}  Y={btn_y}  LB={btn_lb}  RB={btn_rb}")
