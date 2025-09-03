// Jenkinsfile
pipeline {
    agent any

    environment {
        APP_NAME = "projet-synthese-ia"
    }
    parameters {
        choice(name: 'Components', choices: ['API', 'Monitoring',  'Model', 'Prediction', 'MlFlow', 'All'], description: 'Pick component to deploy')
    }

    stages {
        stage('Checkout repo') {
            steps {
                script {
                    git branch: 'main', url: 'https://github.com/florentio/projet-synthese-ia.git'
                }
            }
        }
        stage('Model Training') {
             when {
                expression {  params.Components == 'Model' ||
                             params.Components == 'All'
                             }
            }
            steps {
                script {
                    sh "docker compose up --build -d training"
                }
            }
        }
        stage('Deploy MlFlow') {
            when {
                expression {  params.Components == 'MlFlow' ||
                             params.Components == 'All' }
            }
            steps {
                script {
                    sh "docker compose up --build -d postgres mlflow"
                }
            }
        }
        stage('Deploy API') {
            when {
                expression {
                             params.Components == 'API' ||
                             params.Components == 'All'
                 }
            }
            steps {
                script {
                    sh "docker compose up --build -d fastapi"
                }
            }
        }
        stage('Deploy Prediction') {
            when {
                expression {
                             params.Components == 'Prediction' ||
                             params.Components == 'All'
                 }
            }
            steps {
                script {
                    sh "docker compose up --build -d flask streamlit"
                }
            }
        }
        stage('Deploy Monitoring') {
            when {
                expression {  params.Components == 'Monitoring' ||
                             params.Components == 'All'
                              }
            }
            steps {
                script {
                    sh "docker compose up --build -d monitoring prometheus grafana"
                }
            }
        }


    }
    post {
        always {
            echo 'Pipeline finished.'
        }
        failure {
            echo 'Pipeline failed. Check logs for details.'
        }
        success {
            echo 'Pipeline completed successfully!'
        }
    }
}