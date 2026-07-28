# PrMers v4.20.76-alpha-v99.87 — Gaussian-Mersenne factor lift

## Ajouts

- `-gm` / `-gm-proth` : preuve déterministe de Proth des normes
  `2^p-(2/p)2^((p+1)/2)+1`.
- `-gm-prp` : PRP Fermat rapide.
- lift exact dans `M_(4p)` pour réutiliser Aevum sans modifier ses kernels.
- chaîne d'exponentiation spécialisée de `p` ou `p-2` opérations.
- sélection automatique d'une petite base de Jacobi `-1`.
- crible spécialisé `q=4kp+1`.
- reprise CRC et mode `-gm-safe` par rejeu indépendant de blocs.
- JSON de résultat et tests de non-régression source.

## Isolation

Aucun fichier dans `third_party/aevum`, `kernels/`, les chemins LL/PRP,
P-1 ou ECM n'a été modifié pour cette extension.

## Limite connue

Le lift emploie une transformée pour `2^(4p)-1`. Une future version native
pour le trinôme creux pourra réduire la taille de transformée, mais nécessitera
un backend et une campagne de validation séparés.
