# Extension Gaussian-Mersenne de PrMers v99.88

Cette extension ajoute un chemin **strictement optionnel** pour tester la norme

\[
G_p=N((1+i)^p-1)=2^p-\left(\frac2p\right)2^{(p+1)/2}+1,
\]

avec `p` premier. Aucun kernel Aevum existant n'est modifié et tous les modes
Mersenne, LL, PRP, P-1 et ECM gardent leur code et leur sélection de plans.

## Méthode GPU sûre utilisée dans cette version

Pour `m=(p+1)/2` et `chi=(2/p)`, on a

```text
G_p = 2^p - chi*2^m + 1.
```

Modulo `G_p`, poser `x=2^m` donne `x^2-2*chi*x+2=0`, puis `x^4=-4`.
Comme `G_p` est impair :

```text
2^(2p) = -1 (mod G_p)
2^(4p) =  1 (mod G_p)
G_p divise 2^(4p)-1.
```

PrMers calcule donc l'exponentiation dans l'anneau Mersenne déjà éprouvé
`Z/(2^(4p)-1)Z`, avec le moteur Aevum inchangé, puis projette le résidu final
modulo `G_p`. Cette construction est exacte : ce n'est pas une approximation
ni un PRP fondé sur une identité supposée.

Cette stratégie « factor lift » a deux avantages importants :

* zéro modification des kernels et de la réduction Mersenne existants ;
* accès immédiat aux plans Aevum Type4/FFT323161, aux checkpoints et aux
  contrôles de roundoff existants.

Sa limite est claire : la transformée correspond à l'exposant `4p`. Elle est
donc plus longue qu'un futur backend natif creux pour
`2^p +/- 2^((p+1)/2) + 1`. Cette v99.88 privilégie la correction et l'absence
de régression. Elle ne doit pas être présentée comme un record de vitesse avant
un benchmark réel sur le GPU cible.


## Factorisation P-1 et ECM (v99.88)

La v99.88 ajoute deux modes strictement optionnels qui réutilisent le même lift exact :

```text
-gm-pm1    P-1 Stage 1 + Stage 2 product-exponent
-gm-ecm    ECM Montgomery Suyama Stage 1 + Stage 2 product-exponent
```

La documentation détaillée, les limites de sûreté du lift et les commandes de
validation sont dans [README_GAUSSIAN_FACTORING.md](README_GAUSSIAN_FACTORING.md).

## Preuve déterministe de Proth

Pour `p=2m-1`, la norme est un nombre de Proth :

```text
G_p = k*2^m + 1
k = 2^(m-1) - chi,  k impair, 0 < k < 2^m.
```

Le mode `-gm` choisit une petite base `a` telle que `Jacobi(a/G_p)=-1`, puis
calcule

```text
a^((G_p-1)/2) mod G_p.
```

Si le résidu vaut `-1`, le théorème de Proth certifie **déterministement** que
`G_p` est premier. L'exposant spécial est exécuté avec seulement `p`, ou
`p-2`, opérations GPU selon `p mod 8`; il n'est pas parcouru bit par bit de
façon générique.

Le mode `-gm-prp` calcule `a^(G_p-1) mod G_p`. Un résidu `1` signifie seulement
« probable premier ».

## Sécurité des calculs

* Checkpoints atomiques `.new/.old`, CRC32 et reprise automatique.
* `-gm-safe` rejoue chaque bloc depuis le même état dans deux registres puis
  compare les résultats sur le GPU. Une erreur injectée avec `-erroriter` doit
  être détectée et le bloc est rejoué.
* Le rejeu complet coûte environ deux fois le calcul. Ce n'est pas encore le
  contrôle Gerbicz-Li compact de PRP Mersenne; ce mode est volontairement plus
  simple et plus conservateur.
* Pour un candidat réellement positif, refaire au minimum une seconde preuve
  sur un autre GPU, avec une autre base de Jacobi `-1`, puis une validation par
  un logiciel indépendant.

## Crible spécialisé des facteurs

Tout facteur premier `q != 5` de `G_p` vérifie

```text
q = 4*k*p + 1.
```

`-gm-sieve L` n'énumère donc pas tous les nombres premiers sous `L`; il teste
uniquement les candidats admissibles `4*k*p+1`. Pour les grands exposants, une
borne telle que `10^12` ne représente que quelques milliers de candidats.

## Options

```text
-gm, -gm-proth       preuve déterministe de Proth
-gm-prp              PRP Fermat base a
-gm-base a           impose la base a
-gm-sieve L          cherche les facteurs admissibles q <= L; 0 désactive
-gm-safe             rejeu indépendant complet par blocs
-gm-replay-block N   longueur des blocs de rejeu
-gm-cpu              oracle GMP de référence
-aevum               force le plugin Aevum
-aevum-fft SPEC      force un plan Aevum précis
-d DEVICE            index OpenCL
```

## Construction Ubuntu

Depuis le répertoire PrMers :

```bash
chmod +x scripts/build_gaussian_mersenne_ubuntu.sh
./scripts/build_gaussian_mersenne_ubuntu.sh
```

Le script construit le plugin Aevum embarqué, PrMers, puis exécute les tests
mathématiques et les gardes d'isolation.

Installation système facultative :

```bash
sudo make install
```

L'installation n'est pas nécessaire si le programme est lancé depuis la
racine du dépôt : le chargeur trouve automatiquement
`third_party/aevum/build-engine/libaevum_engine.so`.

## Tests rapides connus

```bash
# Oracle GMP, petits nombres
./prmers 7  -gm -gm-cpu -gm-sieve 0 -d 1
./prmers 13 -gm -gm-cpu -gm-sieve 1000000 -d 1

# Aevum : G_7 = 113 premier
./prmers 7 -gm -aevum -gm-sieve 0 -d 1

# Composé : le crible trouve 53 pour G_13
./prmers 13 -gm -aevum -gm-sieve 1000000 -d 1

# Plus grand petit cas de validation
./prmers 113 -gm -aevum -gm-safe -gm-sieve 0 -d 1

# Vérification de la détection d'erreur par rejeu
./prmers 113 -gm -aevum -gm-safe -erroriter 20 -gm-sieve 0 -d 1
```

## Validation sur le plus grand exposant connu

Le test ci-dessous est une vraie preuve complète mais peut prendre longtemps,
car le lift Aevum utilise l'exposant `4p` :

```bash
./prmers 15317227 -gm -aevum -d 1 \
  -gm-sieve 1000000000000 \
  -t 1800 -f ./gm-results/15317227 \
  2>&1 | tee gm-15317227.log
```

Pour un premier benchmark, lancer le mode PRP sans rejeu :

```bash
./prmers 15317227 -gm-prp -gm-base 3 -aevum -d 1 \
  -gm-sieve 1000000000000 \
  -t 1800 -f ./gm-results/15317227-prp \
  2>&1 | tee gm-prp-15317227.log
```

## Exposants de recherche au-delà de 15 317 227

Ces nombres sont uniquement des **exposants premiers de test**. Ils ne sont pas
présentés comme des plages libres ni comme des candidats jamais calculés : il
faut consulter la coordination du projet avant toute revendication.

```bash
for p in 15317251 15400031 16000057 18000041 20000003; do
  ./prmers "$p" -gm-prp -gm-base 3 -aevum -d 1 \
    -gm-sieve 1000000000000 -t 1800 -f "./gm-results/${p}" \
    2>&1 | tee "gm-prp-${p}.log" || true
done
```

Un PRP positif doit ensuite être certifié :

```bash
./prmers P -gm -aevum -gm-safe -d 1 \
  -gm-sieve 1000000000000 -t 1800 -f "./gm-results/P" \
  2>&1 | tee "gm-proth-P.log"
```

## Résultats produits

```text
gm_prp_p<P>.ckpt
gm_proth_p<P>.ckpt
gm_prp_p<P>_result.json
gm_proth_p<P>_result.json
results.txt
```

Les checkpoints sont supprimés après une fin normale et conservés après une
interruption.

## Validation effectuée avant livraison

* vérification exacte indépendante des identités de lift sur les petits cas ;
* comparaison de la chaîne optimisée avec `pow(a,(G_p-1)/2,G_p)` ;
* cas premiers connus jusqu'à `p=113` dans le test autonome ;
* vérification syntaxique C++20 des trois unités modifiées ;
* garde source confirmant qu'aucun fichier de `third_party/aevum` n'a changé ;
* `git diff --check`.

Le conteneur de construction utilisé pour préparer l'archive ne possédait pas
les en-têtes OpenCL de développement complets. Le lien final GPU n'a donc pas
pu être exécuté ici. Le script fourni effectue cette construction sur l'Ubuntu
cible, où les dépendances OpenCL/GMP sont déjà installées.

## Références

- P. Berrizbeitia et B. Iskra, *Gaussian Mersenne and Eisenstein Mersenne
  Primes*, Mathematics of Computation 79 (2010), 1779–1791,
  DOI `10.1090/S0025-5718-10-02324-0`.
- OEIS A057429, exposants des normes Gaussian-Mersenne premières.
