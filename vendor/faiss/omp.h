#pragma once

typedef int omp_lock_t;

static inline int omp_get_max_threads(void) {
    return 1;
}

static inline int omp_get_num_threads(void) {
    return 1;
}

static inline int omp_get_thread_num(void) {
    return 0;
}

static inline int omp_get_nested(void) {
    return 0;
}

static inline int omp_in_parallel(void) {
    return 0;
}

static inline void omp_set_nested(int) {
}

static inline void omp_set_num_threads(int) {
}

static inline void omp_init_lock(omp_lock_t * lock) {
    *lock = 0;
}

static inline void omp_destroy_lock(omp_lock_t *) {
}

static inline void omp_set_lock(omp_lock_t * lock) {
    *lock = 1;
}

static inline void omp_unset_lock(omp_lock_t * lock) {
    *lock = 0;
}

static inline int omp_test_lock(omp_lock_t * lock) {
    if (*lock) {
        return 0;
    }
    *lock = 1;
    return 1;
}
