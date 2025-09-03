// Jenkinsfile
pipeline {
    agent any

    environment {
        APP_NAME = "projet-synthese-ia"
    }

    stages {
        stage('Deploy') {
            steps {
                script {
                    sh "docker compose up -d --build"
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