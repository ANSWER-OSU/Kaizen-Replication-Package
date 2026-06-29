#pragma omp parallel for
for (int i = 1; i < x_points - 1; i++) {
  for (int j = 1; j < y_points - 1; j++) {
    u_new[idx(i, j)] = u[idx(i, j)] + (nu * del_t / (del_x * del_x)) * (u[idx(i, j + 1)] + u[idx(i, j - 1)] - 2 * u[idx(i, j)]) + (nu * del_t / (del_y * del_y)) * (u[idx(i + 1, j)] + u[idx(i - 1, j)] - 2 * u[idx(i, j)]) - (del_t / del_x) * u[idx(i, j)] * (u[idx(i, j)] - u[idx(i, j - 1)]) - (del_t / del_y) * v[idx(i, j)] * (u[idx(i, j)] - u[idx(i - 1, j)]);
    v_new[idx(i, j)] = v[idx(i, j)] + (nu * del_t / (del_x * del_x)) * (v[idx(i, j + 1)] + v[idx(i, j - 1)] - 2 * v[idx(i, j)]) + (nu * del_t / (del_y * del_y)) * (v[idx(i + 1, j)] + v[idx(i - 1, j)] - 2 * v[idx(i, j)]) - (del_t / del_x) * u[idx(i, j)] * (v[idx(i, j)] - v[idx(i, j - 1)]) - (del_t / del_y) * v[idx(i, j)] * (v[idx(i, j)] - v[idx(i - 1, j)]);
  }
}

#pragma omp parallel for
for (int i = 0; i < x_points; i++) {
  u_new[idx(0, i)] = 1.0;
  v_new[idx(0, i)] = 1.0;
  u_new[idx(y_points - 1, i)] = 1.0;
  v_new[idx(y_points - 1, i)] = 1.0;
}

#pragma omp parallel for
for (int j = 0; j < y_points; j++) {
  u_new[idx(j, 0)] = 1.0;
  v_new[idx(j, 0)] = 1.0;
  u_new[idx(j, x_points - 1)] = 1.0;
  v_new[idx(j, x_points - 1)] = 1.0;
}

#pragma omp parallel for
for (int i = 0; i < n; i++) {
  u[i] = u_new[i];
  v[i] = v_new[i];
}