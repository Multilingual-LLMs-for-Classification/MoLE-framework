// ============================================================================
// MoLE-framework — Jenkins Pipeline
// ============================================================================
//
// Stages:
//   1. Checkout          — clone/update from GitHub
//   2. Build Image       — docker build coordinator image
//   3. Run Tests         — pytest (CPU-only, no Ray/GPU required)
//   4. Deploy Coordinator— SSH → csetuf07 → restart coordinator
//   5. Deploy Workers    — SSH → worker machines in parallel
//   6. Health Check      — verify cluster is healthy after deploy
//
// Required Jenkins credentials (configure in Manage Jenkins → Credentials):
//   csetuf07-ssh-key   — SSH private key for cse@csetuf07 (10.8.100.21)
//   csetuf14-ssh-key   — SSH private key for cse_g4@10.8.100.28
//
// Required Jenkins plugins:
//   - SSH Agent Plugin
//   - Pipeline
//   - Docker Pipeline
// ============================================================================

pipeline {

    agent any

    // -------------------------------------------------------------------------
    // Parameters — allows manual trigger with options
    // -------------------------------------------------------------------------
    parameters {
        booleanParam(
            name: 'DEPLOY_COORDINATOR',
            defaultValue: true,
            description: 'Deploy updated coordinator container to csetuf07'
        )
        booleanParam(
            name: 'DEPLOY_WORKERS',
            defaultValue: true,
            description: 'Deploy updated worker containers to all worker machines'
        )
        booleanParam(
            name: 'SKIP_TESTS',
            defaultValue: false,
            description: 'Skip test stage (use only for hotfix deploys)'
        )
    }

    // -------------------------------------------------------------------------
    // Environment
    // -------------------------------------------------------------------------
    environment {
        COORDINATOR_HOST = '10.8.100.21'
        COORDINATOR_USER = 'cse'
        COORDINATOR_REPO = '/home/cse/Desktop/MoLE-framework'
        COORDINATOR_COMPOSE = 'docker/docker-compose-ray.yml'

        WORKER_14_HOST = '10.8.100.28'
        WORKER_14_USER = 'cse_g4'
        WORKER_14_REPO = '/home/cse_g4/MoLE-framework'
        WORKER_14_COMPOSE = 'docker/docker-compose-ray-worker.yml'
        WORKER_14_NUM_GPUS = '1'

        // Used by test stage
        DATABASE_URL   = 'sqlite:///./test_users.db'
        SERVICE_MODE   = 'coordinator'
        JWT_SECRET_KEY = 'test-secret-key-for-ci-only'

        // Health check endpoint
        HEALTH_URL = "http://${COORDINATOR_HOST}:8000/api/v1/health"
    }

    // -------------------------------------------------------------------------
    // Global options
    // -------------------------------------------------------------------------
    options {
        timeout(time: 30, unit: 'MINUTES')
        disableConcurrentBuilds()                 // prevent overlapping deploys
        buildDiscarder(logRotator(numToKeepStr: '20'))
    }

    // -------------------------------------------------------------------------
    // Trigger: run on every push; feature/* branches only run tests (no deploy)
    // -------------------------------------------------------------------------
    triggers {
        githubPush()
    }

    // =========================================================================
    // STAGES
    // =========================================================================
    stages {

        // ---------------------------------------------------------------------
        // Stage 1 — Checkout
        // ---------------------------------------------------------------------
        stage('Checkout') {
            steps {
                checkout scm
                echo "Branch: ${env.GIT_BRANCH}  Commit: ${env.GIT_COMMIT?.take(8)}"
            }
        }

        // ---------------------------------------------------------------------
        // Stage 2 — SSH Test (temporary — verify SSH connectivity only)
        // ---------------------------------------------------------------------
        stage('SSH Test') {
            steps {
                sshagent(credentials: ['csetuf07-ssh-key']) {
                    sh """
                        ssh -o StrictHostKeyChecking=no \
                            ${env.COORDINATOR_USER}@${env.COORDINATOR_HOST} \
                            'echo "SSH to csetuf07 OK — hostname: \$(hostname)"'
                    """
                }
                sshagent(credentials: ['csetuf14-ssh-key']) {
                    sh """
                        ssh -o StrictHostKeyChecking=no \
                            ${env.WORKER_14_USER}@${env.WORKER_14_HOST} \
                            'echo "SSH to csetuf14 OK — hostname: \$(hostname)"'
                    """
                }
            }
        }

        // ---------------------------------------------------------------------
        // Stage 3 — Build Docker Image (coordinator)
        // ---------------------------------------------------------------------
        stage('Build Image') {
            steps {
                script {
                    echo 'Building coordinator Docker image...'
                    sh """
                        docker build \
                            -f docker/Dockerfile \
                            -t mole-coordinator:${env.BUILD_NUMBER} \
                            -t mole-coordinator:latest \
                            .
                    """
                }
            }
        }

        // ---------------------------------------------------------------------
        // Stage 4 — Run Tests (CPU-only, no Ray or GPU needed)
        // ---------------------------------------------------------------------
        stage('Run Tests') {
            when {
                not { expression { params.SKIP_TESTS } }
            }
            steps {
                script {
                    echo 'Running unit tests inside coordinator image...'
                    sh """
                        docker run --rm \
                            -e DATABASE_URL=${env.DATABASE_URL} \
                            -e SERVICE_MODE=${env.SERVICE_MODE} \
                            -e JWT_SECRET_KEY=${env.JWT_SECRET_KEY} \
                            -e USE_RAY=false \
                            mole-coordinator:${env.BUILD_NUMBER} \
                            pytest tests/ -v \
                                --ignore=tests/test_api.py \
                                --ignore=tests/test_classifier_modes.py \
                                --tb=short \
                                -q
                    """
                }
            }
            post {
                failure {
                    echo 'Tests failed — aborting deploy.'
                }
            }
        }

        // ---------------------------------------------------------------------
        // Stage 5 — Deploy Coordinator
        // ---------------------------------------------------------------------
        stage('Deploy Coordinator') {
            when {
                allOf {
                    anyOf {
                        branch 'main'
                        branch 'feature/jenkins-deployment-pipeline'
                    }
                    expression { params.DEPLOY_COORDINATOR }
                }
            }
            steps {
                sshagent(credentials: ['csetuf07-ssh-key']) {
                    sh """
                        ssh -o StrictHostKeyChecking=no \
                            ${env.COORDINATOR_USER}@${env.COORDINATOR_HOST} \
                            '
                                set -e
                                echo "[deploy] Pulling latest code..."
                                cd ${env.COORDINATOR_REPO}
                                git pull origin main

                                echo "[deploy] Rebuilding and restarting coordinator..."
                                cd docker
                                docker-compose -f docker-compose-ray.yml \
                                    up --build -d --no-deps coordinator

                                echo "[deploy] Coordinator restarted."
                            '
                    """
                }
            }
        }

        // ---------------------------------------------------------------------
        // Stage 6 — Deploy Workers    [pending]
        // Stage 7 — Health Check      [pending]
        // ---------------------------------------------------------------------

    }

    // =========================================================================
    // POST ACTIONS
    // =========================================================================
    post {
        success {
            echo """
            ✅ Pipeline succeeded.
               Branch  : ${env.GIT_BRANCH}
               Commit  : ${env.GIT_COMMIT?.take(8)}
               Build   : #${env.BUILD_NUMBER}
            """
        }
        failure {
            echo """
            ❌ Pipeline failed at stage: ${env.STAGE_NAME}
               Branch  : ${env.GIT_BRANCH}
               Commit  : ${env.GIT_COMMIT?.take(8)}
               Build   : #${env.BUILD_NUMBER}
            Check the console output for details.
            """
        }
        always {
            // Clean up local test DB if created
            sh 'rm -f test_users.db || true'
            // Remove the workspace clone from the Jenkins machine
            cleanWs()
        }
    }
}
