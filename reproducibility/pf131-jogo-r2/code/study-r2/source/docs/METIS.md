# METIS dans PF C++ 1.3.1

## Sélection

```bash
pf_solver --local-refine lj-newton --newton-linear-solver ldl --newton-linear-ordering metis
pf_solver --local-refine lj-newton --newton-linear-solver mumps --newton-linear-ordering metis
```

Ces fragments montrent les sélections ; les commandes exécutables complètes
sont dans `INSTALL_TEST_COMMANDS.md`. METIS n'est pas un solveur numérique :
il détermine l'ordre des variables avant la factorisation LDLᵀ.

`auto` est le défaut historique. `metis` est accepté avec LDL et MUMPS natifs.
Une demande METIS avec CG/MA27, un ordre inconnu ou une compilation sans METIS
provoque une erreur explicite. MA27 reste utilisable avec `auto` et HSL.
Ipopt ne fournit pas d'interface autonome LDL dans ce portage. L'option
`--newton-linear-ordering` ne configure pas le MUMPS interne à Ipopt.

## Matrice et permutation

La Hessienne Lennard–Jones inclut toutes les interactions. Son graphe structurel
est complet, même si certains coefficients sont actuellement nuls. Aucun seuil
numérique ne supprime de coefficients. Les normales exclues, la courbure des
contraintes actives et la régularisation conservent ce même motif structurel.

METIS_NodeND produit `perm[new]=old` et `inverse[old]=new` en base zéro.
La bijection et l'inversion sont vérifiées. METIS 5 reçoit une graine fixe 0.
METIS 4 utilise ses réglages par défaut ; les permutations peuvent différer
entre versions. Le cache est indexé par dimension dans chaque processus.
Un verrou protège uniquement les appels METIS/le cache ; les factorisations
LDL indépendantes continuent à s'exécuter en parallèle.

Pour LDL, `Aperm(i,j)=A(perm[i],perm[j])`, `bperm(i)=b(perm[i])`, puis
`x(perm[i])=xperm(i)`. Le stockage creux conserve tout le triangle inférieur,
y compris ses zéros numériques. Eigen SimplicialLDLT utilise NaturalOrdering
sur cette matrice déjà permutée. L'analyse symbolique est effectuée une fois
par raffinement local et réutilisée lors des changements de valeurs/diagonale.
Les pivots D doivent être finis et strictement positifs avant la résolution.
Le mode LDL `auto` garde Eigen LDLT dense ; comparer `auto` et `metis` compare
donc aussi deux implémentations de factorisation.

Pour MUMPS, `PERM_IN(old)=inverse[old]+1`, `ICNTL(7)=1`, `ICNTL(28)=1`.
Après l'analyse, `INFOG(7)` doit valoir 1, sinon le calcul s'arrête. C'est un
ordre METIS externe fourni à MUMPS ; le code ne demande pas à MUMPS de choisir
lui-même METIS via `ICNTL(7)=5`, qui peut sinon revenir au choix automatique
lorsque METIS n'est pas disponible dans cette bibliothèque MUMPS.
Les adaptations internes de l'arbre et le pivotage numérique restent du
ressort de MUMPS. `PERM_IN` n'est pas une interdiction de pivotage numérique.

Référence : [manuel MUMPS, permutations symétriques, sections 5.6 et 5.6.2](https://sources.debian.org/src/mumps/5.7.3-3/doc/userguide_5.7.3.pdf/).

## Traçabilité

- `--version` expose `metis=0|1` et `metis_api=0|4|5`.
- `options.newton_linear_ordering` indique la sélection demandée.
- `linear_metis_analyses` compte les analyses symboliques ayant utilisé la
  permutation METIS ; ce n'est pas le nombre d'appels à METIS (cache).
- `mumps_user_ordering_analyses` compte celles confirmées par `INFOG(7)=1`.
- `linear_cg_fallbacks` conserve le compteur des replis numériques CG existants.
  Un repli n'est pas présenté comme une factorisation METIS réussie.
- Dans le résultat du témoin Ipopt, les options Newton sont conservées à titre
  de configuration de campagne mais les compteurs natifs sont nuls.

`pf_linear_check --backend ldl|mumps --ordering metis` vérifie résidus,
inertie, rejet singulier, reprise après échec, réutilisation symbolique,
permutation et minima analytiques de deux atomes. Il affiche l'empreinte de
la permutation et `mumps_infog7=1` pour MUMPS.

## Compilation et limites

Utiliser `PF_WITH_METIS=ON`, `PF_METIS_INCLUDE` et `PF_METIS_LIBRARY`.
Ne pas mélanger des en-têtes METIS 4 et une bibliothèque METIS 5 ou l'inverse,
ni des largeurs d'indices différentes. Les distributions Conda cohérentes sont
préférables à un mélange de bibliothèques provenant de plusieurs projets.
Le MUMPS natif reste séquentiel ; le parallélisme entre particules utilise
des processus persistants. LDL utilise les threads OpenMP.

Les valeurs d'objectif, gradient, Hessienne, facteurs et seconds membres restent
en `double` IEEE 754 binary64. Les indices et permutations restent entiers.

METIS ne peut pas rendre creuse une Hessienne structurellement dense. Aucun
gain de performance ni de qualité globale n'est garanti par cette extension.
Elle ne corrige pas le critère d'arrêt de premier ordre du Newton 1.3.0 ni
sa capacité à s'arrêter sur certaines selles. Les résultats de validation
LJ38/LJ76 portent sur des raffinements locaux, pas sur l'optimum global PF.
