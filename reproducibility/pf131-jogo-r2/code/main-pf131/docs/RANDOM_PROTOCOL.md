# Protocole aléatoire et initialisation — 1.2.0

## Graines des runs

Le CLI utilise exactement la convention de la source reçue :

```python
seed_init = int(hashlib.md5(f"{base}_{n}_{m}_0".encode()).hexdigest()[:8], 16)
seed_pf = int(hashlib.md5(f"{base}_{n}_{m}_1000".encode()).hexdigest()[:8], 16)
```

`--seed` est la base (2024 par défaut), `m` commence à 1. L'API
`derive_seed(base,n,m,offset=0)` renvoie un uint32. Elle remplace SplitMix64.
Le CLI consigne base, seed_init et seed_pf. Dans l'API générique `Options.seed`
désigne directement seed_pf ; il n'est pas dérivé une seconde fois par Solver.
`Options.run_seeds`, facultatif, porte les métadonnées de la campagne CLI.

## Deux objets initiaux distincts dans Python

| Objet | Graine | Domaine par défaut | Rôle dans la source fournie |
|---|---|---|---|
| `x0` | seed_init | [-4,4]^(3N) | Stocké dans `self.x0`, mais ne construit pas les particules de `_solve_pf_loop`. |
| Population PF | seed_pf | [-5,5]^(3N) | Départs effectifs de la boucle PF, N0 tirages de vecteurs successifs. |

Le C++ reproduit les deux objets avec les mêmes coordonnées et dans le même
ordre. Il ne remplace pas arbitrairement une particule par x0 : cela changerait
la procédure Python. `initialization.json` enregistre x0 et indique
`x0_used_by_pf: false`. `initial.pfpop` contient les départs réellement utilisés
par PF et par la baseline multistart Ipopt. Aucune optimisation de x0 n'est
ajoutée. Pour une boîte personnalisée, la marge de x0 reste 0.1*(ub-lb).

N0 vaut max(3N+56,50) par défaut dans le CLI : 170 pour LJ38 et 284 pour LJ76.
La génération est dans l'ordre particule puis coordonnée, comme la liste
Python de `_random_box_point(rng)`.

Après génération, le flux PF doit déjà avoir consommé N0*3N uniformes.
Le C++ respecte cet état, y compris avec `--population` : il saute le nombre
correspondant de tirages bruts PCG64 en O(log(N0*3N)). Recharger une population
produite par le programme, avec la même configuration et la même graine,
reproduit donc la continuation. Un fichier arbitraire remplace les coordonnées,
mais réserve le même nombre de tirages initiaux. Il est marqué
`population_source: external-pfpop` ; son contenu n'est pas certifié égal à Python.

## Générateurs et consommation

L'implémentation native adapte les algorithmes de NumPy 2.3.5 :

- SeedSequence scalaire, pool de quatre uint32, génération de quatre uint64 ;
- PCG64 XSL-RR 128/64, et non PCG64DXSM ;
- uniformes float64 sur 53 bits et normales Ziggurat, sans Box–Muller ;
- entiers bornés par la méthode de Lemire, avec le tampon de demi-mot uint32 ;
- `Generator.choice(..., replace=False, shuffle=True)`, y compris sélection
  complète, algorithme de Floyd et branche tail-shuffle pour les grands ensembles.

La sélection de deux partenaires en mutation utilise aussi `choice`.
Les sources et empreintes sont dans `reference/RANDOM_SOURCES.json` et leurs
licences dans `THIRD_PARTY_NOTICES.md`. Le solveur n'appelle pas Python.
GCC/Clang avec entiers 128 bits est requis. Le projet évite fast-math et la
contraction FMA afin de conserver la succession d'arrondis utilisée par NumPy.

Le filtre tire `rng.integers(0,10000)` pour chaque flow avec clustering, puis
le clustering reçoit un générateur indépendant de type NumPy RandomState
(MT19937 32 bits). Le clustering ne consomme donc plus le flux PCG64 du filtre.
Le clustering périodique conserve `random_state=42`, explicitement présent dans
la source Python. Le flux MT19937 est vérifié ; l'intégralité des partitions
K-means/GMM n'est pas certifiée identique à sklearn ou cuML.

Tous ces tirages sont effectués dans le processus coordinateur, avant les
calculs parallèles. Changer le nombre de workers ne change pas les tirages.
Les limites temporelles et les budgets locaux contraignants peuvent changer
les raffinements terminés, donc la trajectoire résultante.

## Fonctions Python sans graine

`island_exchange` et `tune_parameters` créent chacun un `default_rng()` sans
graine. Leur entropie n'est pas enregistrée dans la source fournie. Les graines
seed_init/seed_pf ne suffisent donc pas à rejouer leurs anciens tirages.

Pour conserver le caractère indépendant de ces générateurs tout en rendant le
C++ reproductible, deux sous-séquences sont définies explicitement : PCG64
initialisé avec seed_pf, avancé de 2^48 tirages pour les îlots et de 2^49 pour
l'injection de diversité. Ces états persistent pendant le solve et n'avancent
pas le flux PF principal. La politique figure dans les JSON ; elle n'est pas
présentée comme la reconstruction de l'entropie du Python.
L'injection déclenchée directement par un ESS très faible continue d'utiliser
le flux PF principal, conformément à la source.

## Validation et limites

`tests/numpy_random_parity.py` compare directement les sorties C++ à NumPy et
exécute les méthodes pertinentes extraites par AST du Python archivé. Il vérifie :
les graines de 30 runs LJ38/LJ76, tous leurs x0 et toutes leurs particules
initiales, les tirages bruts, uniformes, normales, entiers, mélanges de lois,
sélections et la continuation après sauvegarde/rechargement. Un problème
quadratique avec un unique groupe DBSCAN isole et vérifie l'ordre des tirages
dans trois itérations réelles du solveur, pour deux graines.

Ces essais ne constituent pas une reproduction complète LJ76. Restent notamment
les différences de précision JAX/double, de partitions de clustering, de solveur
local, de sélection de l'incumbent, et les sources Python non ensemencées.
La mutation Python passée à joblib avec un générateur partagé/sérialisé dépend
également de son backend ; le C++ utilise un flux coordonné en ordre fixe.

La provenance générale reste celle de l'archive transmise. Cette correction
ne certifie pas que cette archive est la version scientifique ayant obtenu
la campagne LJ76 revendiquée, et ne modifie pas implicitement l'arrêt à 400.
