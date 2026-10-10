# Addendum de protocole : v3.1.0, raffinement local

La v3.0 et ses résultats restent archivés. Cet addendum remplace uniquement la politique de raffinement local de BH-OJ et GA et son interprétation. Le protocole général original reste dans `PROTOCOL_BASELINES_V3.md` pour les éléments inchangés.

## Cause et modification

Le wrapper v3.0 interprétait `res.success` comme `converged`. Avec des coordonnées très répulsives, l'essai initial de L-BFGS-B pouvait pousser plusieurs atomes aux mêmes coins de la boîte; un retour au départ avec variation nulle déclenchait alors `ftol` malgré un gradient gigantesque.

Pour chaque appel L-BFGS-B, la v3.1 fixe un facteur positif

`s = max(1, max(abs(g0))/min(0.1, 0.1*r_min))`

avec plancher numérique `1e-8` sur le dénominateur. Les valeurs fournies au solveur sont `E/s` et `g/s`, tandis que le compteur, les meilleurs points, les sorties et l'audit utilisent toujours **E non redimensionnée**. Les bornes physiques restent celles du protocole. Les critères transmis sont `ftol/s` et `gtol/s`; ils ne sont pas relâchés.

Le facteur est fixé pendant chaque appel; il n'est pas modifié dans une paire énergie/gradient. Il vise à éviter le pas initial excessif. Il ne borne pas tous les pas quasi-Newton ultérieurs. Après une sortie à grand résidu projeté (>1e-3), un pas de gradient projeté avec recherche d'Armijo peut relancer le solveur, au plus huit fois, dans le **même budget total et la même limite d'itérations locales**. Ces reprises sont bornées et sans tirage aléatoire. Toute énergie de récupération est comptée en `objective_local`.

Le seuil 1e-3 décide seulement de ces récupérations. Il ne remplace pas `local_gtol=1e-10` et n'est jamais utilisé pour déclarer la stationnarité à cette tolérance.

## Statuts

- `stationary`: le résidu `max(abs(x-clip(x-g,-R,R)))` du point effectivement retourné est <= `local_gtol`; cela n'est ni un certificat de minimum local ni d'optimalité globale.
- `function_tolerance`: le solveur a arrêté sur son critère fonctionnel; le résidu demandé de stationnarité n'est pas satisfait.
- `nonstationary_exit`: autre sortie du solveur, sans affirmation de convergence.
- `unresolved_nonstationary`: grand résidu persistant après récupérations/limite locale; le meilleur point évalué reste enregistré, et l'anomalie reste explicitement visible.
- `evaluation_limit`, `target_reached`, `time_limit`, `interrupted`: interruptions transmises au moteur commun, avec diagnostics locaux potentiellement incomplets.

Chaque run contient `local_diagnostics_summary`, des exemples détaillés et `local_anomaly_examples`. Les exemples sont bornés à 30 pour ne pas gonfler les fichiers, les totaux ne sont pas tronqués. Le nombre de reprises, d'évaluations de sauvegarde, d'appels redimensionnés, de résidus élevés et de sorties sans mouvement est comptabilisé.

## Éléments inchangés

Cinq classes présentes; seuls BH-OJ/GA appellent la routine modifiée. Potentiel pf131, précision double, domaine, initialisation, seeds, règles de mouvement global, budgets, cible et tolérance de succès sont inchangés. Les AST des classes et les trajectoires techniques non locales sont comparés à v3.0 dans `validation/nonlocal_parity.json`. Le fichier de budgets est identique octet pour octet.

Les résultats v3.0 de PSO/CLPSO/SMC restent utilisables sous leur version d'origine sous réserve des audits déjà prévus. Les résultats v3.0 de BH-OJ/GA ne sont pas supprimés mais doivent être séparés de la comparaison principale utilisant le raffinement corrigé. Ce remplacement s'applique à tous les runs de ces deux méthodes, pas seulement aux échecs, pour éviter une sélection favorable.

Les essais joints sont des validations techniques sur cette machine, pas une nouvelle estimation des taux de succès. Aucun test n'exige l'atteinte de la cible globale LJ38 par une baseline.
