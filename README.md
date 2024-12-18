# MLOPS - Face Recognition

### Abel Andry - Emile Merle - Julien Schaffauser

Ce projet propose une solution complète de reconnaissance faciale basée sur un modèle **OpenCV Cascade** pour la détection des visages, et un **CNN custom** pour la reconnaissance des visages.

## **Fonctionnalités principales**

- **Détection et reconnaissance faciale** :
  - Utilisation d'OpenCV pour détecter les visages.
  - Identification des visages grâce à un modèle de réseau de neurones convolutifs (CNN).
- **Application Kivy** :
  - L'application, autonome, permet d'interagir avec la caméra pour détecter et ajouter des visages.
  - Disponible dans le dossier `app/`.
  - Peut être lancée via le fichier `camerapp.py`.
- **Gestion serveur** :
  - Le serveur est hébergé sur une machine virtuelle Azure.
  - Automatisation du déploiement via GitHub Actions.
  - Mise à jour dynamique du modèle lors de l'ajout de nouvelles personnes.

## **Workflow général**

1. **Ajout d'une nouvelle personne depuis l'application Kivy** :
   - L'utilisateur capture le visage d'une personne via l'application.
   - L'application envoie les données au serveur.

2. **Réentraînement du modèle** :
   - Le serveur lance un entraînement du modèle CNN pour inclure la nouvelle personne.
   - Les fichiers de données et métadonnées sont mis à jour.

3. **Mise à jour Git** :
   - Le serveur publie les nouvelles données sur une branche dédiée `new-model-updates` du dépôt Git.

4. **Automatisation GitHub Actions** :
   - Mise à jour de l'image Docker avec le nouveau modèle.
   - Déploiement automatique de l'image sur la machine virtuelle Azure.


