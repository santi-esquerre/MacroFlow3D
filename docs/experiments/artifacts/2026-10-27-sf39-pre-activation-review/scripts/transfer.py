# SF-39 transfer rule: narrow-ladder and production-twin relative 1/2->1 differences d1,d2 (noisy estimates)
# pass iff same sign and ratio d1/d2 in [1/2, 2].  Pass probability vs true effects and SEs.
import numpy as np
rng=np.random.default_rng(1)
def p(m1,m2,s1,s2,n=400000):
    d1=rng.normal(m1,s1,n); d2=rng.normal(m2,s2,n)
    r=d1/d2; return np.mean((np.sign(d1)==np.sign(d2))&(r>=0.5)&(r<=2))
for (m1,m2,s1,s2,lab) in [(0,0,1,1,'no effect, equal SE'),(0,0,0.056,0.037,'no effect, SE .056/.037'),
    (0.05,0.05,0.056,0.037,'true 5% both'),(0.05,0.05,0.02,0.02,'true 5%, SE .02'),(0.05,0.05,0.01,0.01,'true 5%, SE .01'),
    (0.10,0.0,0.056,0.037,'narrow 10%, twin 0'),(0.01,0.01,0.01,0.01,'true 1%, SE 1%')]:
    print(f"{lab:28s} pass={p(m1,m2,s1,s2):.3f}")
