# Ratios quoted in the article, from the notebook 06 runs on 2026-11-01.
fused=[14.7,15.1,15.7]; eager=[5.0,4.8]; comp=[33.5,33.4]
final=(15.1,4.8,33.5)
print(f"final run: Rust fused over compile, streamed {final[0]/final[2]:.2f}x")
print(f"final run: Rust fused over eager, streamed {final[0]/final[1]:.2f}x")
print(f"final run: compile over Rust fused, in batches {45.1/15.2:.2f}x")
print(f"peak memory, compile over Rust fused {16.91/7.75:.2f}x")
print(f"32,761-token read, Rust fused over compile {8.0/5.0:.2f}x slower")
