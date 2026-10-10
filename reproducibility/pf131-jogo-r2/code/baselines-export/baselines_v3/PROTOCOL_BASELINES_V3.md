# Protocole corrigé v3.0.0 — JOGO, huit tailles et cinq baselines

## 1. Périmètre et budget

Méthodes : BH-OJ, GA, PSO global-best, CLPSO, tempered resample–move SMC.
Tailles : 38,75,76,77,98,102,103,104 ; indices de run 1..30 ; 1200 jobs.
Ce sont les implémentations et paramétrages de ce paquet, pas une affirmation de
supériorité ou de représentation exhaustive de chaque famille.

| N | Maximum HPF communiqué | Budget de chaque baseline |
|---|---:|---:|
|38|486736|973472|
|75|13928846|27857692|
|76|19073098|38146196|
|77|28629700|57259400|
|98|41316611|82633222|
|102|4542882|9085764|
|103|4791940|9583880|
|104|9670017|19340034|

Règle conservée de v2 : deux fois le maximum **observé** HPF fourni par l'auteur.
Ce n'est ni un budget identique entre HPF et baselines, ni une mesure universelle du
coût d'une famille. Une fraction d'appels `objective_pf` n'est pas une fraction de temps.

## 2. Oracle, unités et domaine (différence importante avec v2)

Le fichier C++ joint par l'auteur `pf_cpp_v1.3.1/src/problem.cpp` contient :

```
q = 1/(r2 + 1e-8)
q3 = q*q*q
pair_energy = q3*(q3-2)
gradient_factor = 12*q*q3*(1-q3)
```

C'est le modèle par défaut `pf131` commun aux cinq méthodes. Il utilise la profondeur
unitaire et une convention de longueur différente du LJ `4*(r^-12-r^-6)` à sigma=1,
ainsi qu'une régularisation du carré de distance. Le mode `--potential standard`
reste disponible explicitement mais doit être traité comme une campagne différente.
La copie source et son origine se trouvent dans `archive/PF131_SOURCE_CONVENTION.json`.
Aucune option du solveur C++ non fournie n'est déduite de cette formule.

Domaine des candidats : boîte [-5,5]^(3N) par défaut. Ni pénalité sphérique, ni
contrainte dure de séparation n'est ajoutée. Les initialisations sont réellement
uniformes dans la boîte, sans centrage ni contraction par défaut. `sphere` ou
`--center` sont des changements explicites de protocole pour les méthodes compatibles,
pas une reproduction silencieuse de v2. SMC exige box/sans centrage pour sa cible.
Si HPF charge une population externe via `--population`, la même loi nominale et les
mêmes seeds ne prouvent pas l'égalité des populations ; fournir les fichiers pour un
appariement exact. Les tailles de populations différentes empêchent aussi de parler
d'ensembles initiaux identiques sans protocole dédié.

## 3. Comptage et meilleure configuration

Un seul `EvaluationCounter` appartient au run et survit aux redémarrages. Chaque
évaluation d'énergie de recherche est comptée, locale incluse. Un lot de B candidats
compte B appels, non un seul. À la limite, seul le préfixe encore autorisé est évalué.
Tous les candidats achevés mettent à jour le meilleur point avant tout arrêt ou rejet.
Les doublons réévalués comptent de nouveau ; aucun cache caché n'est utilisé.

Les callbacks analytiques de gradient ont un compteur séparé et ne sont pas ajoutés
au budget d'énergie. Le calcul analytique d'une dérivée n'appelle pas l'oracle énergie.
Il n'y a pas de Hessienne dans ces baselines. Cette séparation correspond aux catégories
du source PF131 fourni ; le lien exact entre ce source et les 240 JSON reste une
vérification de provenance de la campagne, pas un résultat de ces tests.

Un dernier audit scalaire/Kahan recalcule l'énergie du meilleur point : **un appel
d'audit séparé**, sans nouvelle recherche. Il est enregistré hors du budget, avec son
temps, son énergie et la différence avec l'oracle de recherche. L'énergie publiée est
celle de cet audit. Un désaccord supérieur à 1e-8 + 1e-12*abs(E), des coordonnées
invalides ou une énergie sous la référence de plus que la tolérance rendent le run
invalide et bloquent l'export final. Les compteurs sont réservés avant l'appel ; un
appel d'oracle échoué laisse un run d'erreur, non une donnée valide de comparaison.

Les calculs sont float64. Les sommes vectorisées CPU/GPU peuvent différer de quelques
unités d'arrondi de la somme compensée ; aucune égalité bit à bit entre matériels
n'est revendiquée. Le découpage de lots par défaut est 32. Après une cible atteinte,
un lot déjà engagé est achevé et compté. `first_target_eval` indique le premier
franchissement dans l'ordre des lignes ; ce n'est pas un timestamp CUDA par candidat.

## 4. Raffinement réel L-BFGS-B

SciPy `minimize(method='L-BFGS-B')` reçoit le gradient correct et les bornes explicites.
Tolérances ftol=1e-12, gtol=1e-10, maxls=30, maxcor=10. BH-OJ : maxiter=4000,
raffinement après saut limité à 1200 ; GA : maxiter=5000. `ftol` n'est pas une
borne absolue sur l'écart au minimum global ; `gtol` est le test du gradient projeté
utilisé par L-BFGS-B. Les statuts, messages et itérations locales achevées sont tracés.
Une interruption pendant une résolution peut empêcher SciPy de retourner son nombre
final d'itérations ; les appels d'oracle restent cependant comptés exhaustivement.

Pas de clipping/centrage après calcul d'une énergie à laquelle on associerait des
coordonnées transformées. Le raffinement renvoie son meilleur couple effectivement
évalué ; un statut non convergé n'est pas transformé en succès local, mais les points
valides déjà évalués ne sont pas perdus. Aucun fallback IPOPT n'est ajouté aux baselines.

## 5. Paramètres et modifications des méthodes

Les valeurs définitives sont dans `DEFAULTS` + `protocol.method_params`, et sont copiées
dans chaque JSON. Les populations PSO/CLPSO/SMC sont min(300,3N+56) ; GA garde 300.
Les plafonds internes sont 10^6 itérations/générations et ne remplacent pas le budget.

- BH-OJ : Metropolis T=1, pas gaussien .5, stagnation60, 40 sauts non relaxés de pas3.5,
  contraction .85 puis raffinement ; après deux cycles infructueux, réinitialisation
  interne avec contraction .75. Le meilleur point externe n'est jamais écrasé.
- GA : tournoi3, élitisme5%, croisement SBX eta_c15/proba .7, mutation polynomiale
  bornée eta_m20/proba .1 par coordonnée, raffinement des élites toutes les25 générations.
- PSO : w=.7298, c1=c2=1.49618, vitesses initiales normales d'écart-type .1, bornes par
  clipping, mise à jour des meilleurs points individuels et global sur leurs énergies.
- CLPSO : c=1.49445, probabilités d'apprentissage par dimension .05 à .50, refresh7,
  exemplaires représentés par indices de pbest et au moins une dimension étrangère.
  Par défaut w passe de .9 à .4 selon la **fraction du budget consommée** ; l'ancien
  calendrier sur un million de tours était pratiquement constant. `iterations` reste
  une option explicitement journalisée. Ce changement n'est pas une identité avec v2.
- SMC : prior uniforme, beta_min=0 par défaut, beta_max=5*log(1+N), cible ESS=.7*Np,
  deux mouvements MH par niveau, sigma=.6 avec plancher relatif .15. Un beta_min positif
  demandé est réellement appliqué, avec pondération du prior avant mouvement.

### SMC, détail de la réparation

Chaque étape choisit beta suivant l'ESS des **mêmes poids en logarithmes** ensuite
utilisés, rééchantillonne systématiquement, puis effectue les mouvements. Les poids
repartent uniformes avant l'étape suivante. On n'attend pas un seuil ESS=.5 alors que
l'adaptation maintient .7. Aucun écart d'énergie n'est tronqué arbitrairement à700.
La recherche de beta utilise une variable sans dimension, sans incrément forcé1e-6.
Un avancement `nextafter` pour arrondi est compté dans les diagnostics.

Les propositions gaussiennes MH sont réfléchies sur la boîte, sans centrage ni
clipping à masse atomique sur la frontière. Le noyau réfléchi symétrique permet le
rapport d'acceptation énergétique utilisé. À beta_max, la dernière étape inclut encore
ses mouvements, puis la tentative se termine. Un redémarrage reprend du prior avec
un seed distinct et le budget restant. Pas de boucle de rééchantillonnage sans nouveaux
appels MCMC. `mcmc_proposals` compte aussi les préfixes interrompus ; le nombre accepté
porte sur les balayages achevés, identifiés séparément.

## 6. Seeds, reprises et arrêts

Seed du run : entier des 8 premiers hexadécimaux de
`MD5(f'{base}_{N}_{run}_{1000}')`, base2024, run1..30, sans masque signé.
Le premier optimiseur utilise **directement** ce seed. Tentative a>=2 :
`MD5(f'{run_seed}_{N}_{a}_{1000}')[:8]`. Toutes sont enregistrées. Le meilleur point global reste disponible : en particulier, les sauts BH-OJ peuvent utiliser ce meilleur point. Des seeds distincts ne signifient pas des tentatives statistiquement indépendantes conditionnellement à cette mémoire.
RNG : NumPy Generator/PCG64 côté CPU, y compris si les lots énergie sont sur GPU.

Pas de prétention qu'un hash prouve une indépendance statistique absolue, ni que des
appels RNG consommés différemment donnent les mêmes trajectoires d'une méthode à l'autre.

Arrêts : target_reached, evaluation_limit, time_limit, restart_limit,
single_attempt_complete, numerical_error, audit_failed, interrupted. Priorité aux
interruptions explicites, au franchissement enregistré de cible, au budget, puis au
délai lorsque plusieurs conditions sont simultanées. La tolérance commune vaut1e-4
et le succès est recalculé sur l'énergie auditée (`abs(E-reference)<=tol`). Les références
arrondies sont les valeurs fournies dans v2, pas de nouveaux certificats mathématiques.
Un franchissement non confirmé à l'audit devient audit_failed, non un faux succès.

Plafond temps global :36000s, coopératif ; aucune échéance par tentative par défaut.
Un --restart-seconds positif est un vrai changement de protocole. Une tentative
numériquement erronée n'est pas redémarrée silencieusement. Les limites de temps et
de nombre d'essais sont des censures distinctes de la limite d'évaluations.

Le code, les budgets, options effectives, dépendances, matériel visible et concurrence
sont empreintés. Une reprise ne saute qu'un résultat identique et auto-vérifié. Un
run d'erreur exige --force et archivage ; pas de reprise inexacte au milieu du RNG.

## 7. Validation et reporting

Le résumé final exige toutes les identités attendues, des résultats uniques, des
énergies cohérentes avec les coordonnées, des compteurs cohérents et un protocole
homogène. Les fichiers absents, malformés, dupliqués et les erreurs ne sont jamais
transformés implicitement en réussites ou omis. `--allow-censored` autorise seulement
les censures déclarées et conserve leurs effectifs séparés dans les tableaux.
Les descriptifs énergie/temps couvrent tous les runs valides, succès et échecs.
Les médianes moyennent les deux valeurs centrales pour30 ; SD utilise n-1.

Une comparaison HPF vs baselines doit ensuite rapprocher les populations/configurations,
le modèle et le comptage exacts de la campagne C++. Les résultats n'établiront que la
performance de ces implémentations sous ce protocole, pas celle des familles en général.
Pas d'extrapolation à d'autres problèmes que les instances effectivement calculées.

## Sources de spécification

- Code utilisateur `pf_cpp_v1.3.1_source.tar.gz`, `src/problem.cpp` et callbacks
  d'énergie/dérivées : source primaire du modèle PF131 et des catégories de compteurs.
- Code utilisateur `baselines_v2.tar.gz` : familles, mécanismes et budgets de départ.
- Documentation SciPy consultée pour le contrat L-BFGS-B :
  https://docs.scipy.org/doc/scipy/reference/optimize.minimize-lbfgsb.html
- Documentation CuPy pour conversion et dtype :
  https://docs.cupy.dev/en/stable/reference/generated/cupy.asarray.html

Ces documentations ne constituent pas une validation du bundle : les tests locaux
et le précontrôle sur la machine cible sont consignés séparément.
