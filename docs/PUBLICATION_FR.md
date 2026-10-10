# Téléversement vers francoisakoa/hpfo

Le dossier local `hpfo` contient tout le matériel assemblé. Les archives de données
sont dans `release-assets/`, volontairement exclu de Git. Les sources, scripts,
notices, tables et manifestes sont dans l'arbre destiné à Git.

## Préparer puis vérifier

```bash
python3 scripts/verify_package.py --assets
python3 scripts/build_and_test.py --jobs 4 --native
python3 scripts/verify_package.py --for-publish
python3 scripts/publish_github.py
```

La dernière commande affiche le plan sans accès réseau ni publication.
Avant l'étape suivante, examiner les notices de licence, les chemins et les données
personnelles éventuelles. Le contrôle automatique de secrets est limité.
Ne pas joindre les identifiants Editorial Manager, les commentaires nominatifs des
reviewers, les clés API, les environnements Conda ou les bibliothèques HSL propriétaires.

## Publication explicite

Git et GitHub CLI `gh` doivent être installés. Le dépôt `francoisakoa/hpfo` doit déjà
exister et le compte connecté doit avoir l'autorisation d'y écrire.

```bash
gh auth login
gh auth setup-git
python3 scripts/publish_github.py --apply --confirm-rights-and-privacy
```

Le script clone le dépôt existant dans un répertoire de travail séparé, préserve
son contenu, crée une **nouvelle branche** `review/pf131-jogo-r2`, ajoute les fichiers,
et crée le tag `pf131-jogo-r2`. Il refuse les conflits de fichiers hors fusion
explicite du README/.gitignore ; la citation globale existante est conservée.
Il ne force aucun push, ne supprime rien et ne remplace pas la branche par défaut.

Les données sont téléversées vers une Release d'abord en brouillon ; les noms,
tailles et empreintes distantes disponibles sont vérifiés avant de la publier.
Si le script s'arrête pendant cette étape, ne pas présenter la Release comme prête.
Un brouillon n'est pas un lien de données adéquat pour des reviewers en lecture seule.

Les liens imprimés à la fin désignent précisément le tag et la Release.
Le script ne change pas la visibilité du dépôt. Si le dépôt est privé, accorder
l'accès aux reviewers/à l'éditeur avant de transmettre ces liens.
Ce dépôt identifié par le nom de l'auteur n'est pas anonyme.

## Reprise après erreur

Le dossier TRX initial, les campagnes et les archives source sont conservés.
Les répertoires `.local/` contiennent les diagnostics. Une branche, un tag ou un
répertoire `hpfo` déjà existant ne sera pas écrasé automatiquement. Ne pas utiliser
`--force` pour masquer un conflit : examiner le diagnostic d'abord.
