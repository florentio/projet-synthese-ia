// Jenkinsfile
pipeline {
    agent any

    environment {
        MLFLOW_MODEL_NAME = 'ProjetAIModel'
        APP_NAME = "projet-synthese-ia"
    }

    stages {
        stage('Checkout') {
            steps {
                git branch: 'main', url: 'https://github.com/florentio/projet-synthese-ia.git'
            }
        }
      /*   stage('DVC Pull') {
            steps {
                script {
                    sh 'dvc pull' // Pull latest data and models
                }
            }
        }*/
        stage('Install Dependencies') {
            steps {
                sh 'pip install -r requirements.txt'
            }
        }

/*         stage('Train and Register Model') {
            steps {
                sh "python3 src/train_model.py"
            }
        } */

        stage('Build Docker Images') {
            steps {
                script {
                    sh "docker build -t ${APP_NAME}-api:${BUILD_NUMBER} -f Dockerfile.api ."
                    sh "docker build -t ${APP_NAME}-streamlit:${BUILD_NUMBER} -f Dockerfile.streamlit ."
                }
            }
        }
//         stage('Deploy to Production') {
//             when {
//                 expression { return currentBuild.result == 'SUCCESS' && input(id: 'prod_deploy', message: 'Proceed to Production Deployment?', ok: 'Deploy') }
//             }
//             steps {
//                 script {
//                     sh "mlflow.pyfunc.model.set_production_stage(model_name='ProjetAIModel', version=${env.MLFLOW_MODEL_VERSION}, stage='Production')"
//                 }
//                 sh """
//                 docker-compose up -d --no-deps api streamlit
//                 """
//             }
//         }
//         stage('Run EvidentlyAI Monitoring (Staging)') {
//             steps {
//                 sh "python3 src/monitoring/evidently_metrics.py"
//                 archiveArtifacts artifacts: 'src/monitoring/evidently_report.html'
//             }
//         }
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