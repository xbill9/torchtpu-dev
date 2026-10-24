"""Ratios the article derives from the notebook outputs in this directory. Run it to regenerate arithmetic.txt."""
rows = [
    ("4-bit table over compiled step, run 1 (50.4 / 33.4)", 50.4 / 33.4),
    ("4-bit table over compiled step, run 2 (50.2 / 31.3)", 50.2 / 31.3),
    ("JAX bf16 over 4-bit-table step, run 1 (131.2 / 50.4)", 131.2 / 50.4),
    ("JAX bf16 over 4-bit-table step, run 2 (132.9 / 50.2)", 132.9 / 50.2),
    ("32,761-token read, 131,328 over 33,024 context, run 3 (3305 / 6601)", 3305 / 6601),
    ("32,761-token read, 131,328 over 33,024 context, run 1 (3692 / 6618)", 3692 / 6618),
]
with open(__file__.replace(".py", ".txt"), "w") as f:
    for label, value in rows:
        f.write(f"{label}: {value:.2f}x = {value:.0%}\n")
