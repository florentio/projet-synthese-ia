// Jenkinsfile
pipeline {
    agent any

    environment {
        APP_NAME = "projet-synthese-ia"
    }
    parameters {
        choice(name: 'Components', choices: ['API', 'Monitoring',  'Model Training', '', 'MlFlow', 'DVC', 'All'], description: 'Pick component to deploy')
    }

    stages {
        stage('Checkout repo') {
            steps {
                script {
                    git branch: 'main', url: 'https://github.com/florentio/projet-synthese-ia.git'
                }
            }
        }
        stage('DVC Pull') {
             when {
                expression {  params.Components == 'DVC' ||
                             params.Components == 'All'
                             }
            }
            steps {
                script {
                    sh 'dvc pull' // Pull latest data and models
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
                    sh '''
                    echo 'Model Training'
                    '''
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
                    sh "docker-compose up -d mlflow"
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
                    sh "docker build -t ${APP_NAME}-api:${BUILD_NUMBER} -f Dockerfile.api ."
                    sh "docker build -t ${APP_NAME}-streamlit:${BUILD_NUMBER} -f Dockerfile.streamlit ."
                    sh "docker-compose up -d --no-deps api streamlit"
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
                    sh "docker-compose up -d prometheus grafana"
                }
            }
        }

        stage('EvidentlyAI') {
             when {
                expression {  params.Components == 'EvidentlyAI' ||
                             params.Components == 'All'
                            }
            }
            steps {
                script {
                    sh '''
                    echo 'EvidentlyAI'
                    '''
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