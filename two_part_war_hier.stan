data {
  int<lower=1> N;                     // number of players
  int<lower=1> K;                     // number of predictors (no year here)
  int<lower=1> T;                     // number of draft years (2017,2018 => T=2)

  matrix[N, K] X;                     // predictors (shared by both parts)
  int<lower=0, upper=1> z[N];         // reached MLB within 7 years (0/1)
  vector[N] y;                        // WAR_7yr (0 for non-MLB, real for MLB)

  int<lower=1, upper=T> year_id[N];   // year index for each player
}

parameters {
  // logistic (reach) coefficients
  vector[K] beta_Z;

  // continuous (WAR | reach) coefficients
  vector[K] beta_Y;
  real<lower=0> sigma_Y;

  // hierarchical year effects for reach
  real mu_alpha_Z;
  real<lower=0> tau_alpha_Z;
  vector[T] alpha_Z_raw;

  // hierarchical year effects for WAR | reach
  real mu_alpha_Y;
  real<lower=0> tau_alpha_Y;
  vector[T] alpha_Y_raw;
}

transformed parameters {
  vector[T] alpha_Z;
  vector[T] alpha_Y;

  alpha_Z = mu_alpha_Z + tau_alpha_Z * alpha_Z_raw;
  alpha_Y = mu_alpha_Y + tau_alpha_Y * alpha_Y_raw;
}

model {
  vector[N] eta_Z;
  vector[N] mu_Y;

  // Priors

  // Reach-MLB coefficients (log-odds scale)
  beta_Z      ~ normal(0, 1.5);

  // WAR|reach coefficients (WAR scale, stronger shrinkage)
  beta_Y      ~ normal(0, 0.5);

  // Residual SD for WAR|reach (updated)
  sigma_Y     ~ exponential(0.5);

  // Hyperpriors for year effects
  mu_alpha_Z  ~ normal(0, 1);
  mu_alpha_Y  ~ normal(0, 2);

  tau_alpha_Z ~ exponential(1);
  tau_alpha_Y ~ exponential(1);

  // Standardized year effects
  alpha_Z_raw ~ normal(0, 1);
  alpha_Y_raw ~ normal(0, 1);

  // Linear predictors
  for (i in 1:N) {
    eta_Z[i] = alpha_Z[year_id[i]] + X[i] * beta_Z;
    mu_Y[i]  = alpha_Y[year_id[i]] + X[i] * beta_Y;
  }

  // Reach likelihood
  z ~ bernoulli_logit(eta_Z);

  // WAR likelihood: only for players who reached MLB
  for (i in 1:N) {
    if (z[i] == 1) {
      y[i] ~ normal(mu_Y[i], sigma_Y);
    }
  }
}
