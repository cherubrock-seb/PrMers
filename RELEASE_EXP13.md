# exp13 publication commands

Publish only after both exact GPU differential markers pass.

## Aevum

```
AEVUM_TAG=v0.3.68-throughput-auto-pfa-bridge-exp13
git tag -a "$AEVUM_TAG" -m "Aevum v0.3.68 throughput auto and PFA9 bridge"
git push origin "$AEVUM_TAG"
```

## PrMers

```
TAG=v4.20.69-alpha-v99.75-aevum-throughput-auto-pfa-bridge-exp13
git tag -a "$TAG" -m "PrMers v99.75 throughput auto and PFA9 bridge"
git push origin "$TAG"
```
