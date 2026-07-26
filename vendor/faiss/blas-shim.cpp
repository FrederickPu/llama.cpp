#include <cctype>

// Minimal BLAS/LAPACK symbols for the vendored FAISS build used by premise-server.
// The premise index only uses IndexFlatIP; LAPACK-only FAISS APIs are not supported here.

static bool is_transposed(const char * value) {
    return value && std::toupper((unsigned char) value[0]) == 'T';
}

template <typename T>
static T matrix_value(const T * a, int lda, int row, int col, bool transposed) {
    return transposed ? a[col + row * lda] : a[row + col * lda];
}

template <typename T>
static int gemm_impl(const char * transa, const char * transb, int * m, int * n, int * k,
        const T * alpha, const T * a, int * lda, const T * b, int * ldb, T * beta, T * c, int * ldc) {
    const bool ta = is_transposed(transa);
    const bool tb = is_transposed(transb);

    for (int col = 0; col < *n; ++col) {
        for (int row = 0; row < *m; ++row) {
            T sum = 0;
            for (int i = 0; i < *k; ++i) {
                sum += matrix_value(a, *lda, row, i, ta) * matrix_value(b, *ldb, i, col, tb);
            }
            c[row + col * *ldc] = *alpha * sum + *beta * c[row + col * *ldc];
        }
    }
    return 0;
}

extern "C" int sgemm_(const char * transa, const char * transb, int * m, int * n, int * k,
        const float * alpha, const float * a, int * lda, const float * b, int * ldb, float * beta, float * c, int * ldc) {
    return gemm_impl(transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
}

extern "C" int dgemm_(const char * transa, const char * transb, int * m, int * n, int * k,
        const double * alpha, const double * a, int * lda, const double * b, int * ldb, double * beta, double * c, int * ldc) {
    return gemm_impl(transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
}

extern "C" int sgemv_(const char * trans, int * m, int * n, float * alpha, const float * a, int * lda,
        const float * x, int * incx, float * beta, float * y, int * incy) {
    const bool t = is_transposed(trans);
    const int rows = t ? *n : *m;
    const int cols = t ? *m : *n;
    for (int row = 0; row < rows; ++row) {
        float sum = 0.0f;
        for (int col = 0; col < cols; ++col) {
            sum += matrix_value(a, *lda, row, col, t) * x[col * *incx];
        }
        y[row * *incy] = *alpha * sum + *beta * y[row * *incy];
    }
    return 0;
}

extern "C" int ssyrk_(const char *, const char * trans, int * n, int * k, float * alpha,
        float * a, int * lda, float * beta, float * c, int * ldc) {
    const bool t = is_transposed(trans);
    for (int col = 0; col < *n; ++col) {
        for (int row = 0; row < *n; ++row) {
            float sum = 0.0f;
            for (int i = 0; i < *k; ++i) {
                sum += matrix_value(a, *lda, row, i, t) * matrix_value(a, *lda, col, i, t);
            }
            c[row + col * *ldc] = *alpha * sum + *beta * c[row + col * *ldc];
        }
    }
    return 0;
}

static int lapack_error(int * info) {
    if (info) {
        *info = -1;
    }
    return 0;
}

extern "C" int sgeqrf_(int *, int *, float *, int *, float *, float * work, int * lwork, int * info) {
    if (work && lwork && *lwork == -1) {
        work[0] = 1.0f;
    }
    return lapack_error(info);
}

extern "C" int sorgqr_(int *, int *, int *, float *, int *, float *, float * work, int * lwork, int * info) {
    if (work && lwork && *lwork == -1) {
        work[0] = 1.0f;
    }
    return lapack_error(info);
}

extern "C" int ssyev_(const char *, const char *, int *, float *, int *, float *, float * work, int * lwork, int * info) {
    if (work && lwork && *lwork == -1) {
        work[0] = 1.0f;
    }
    return lapack_error(info);
}

extern "C" int dsyev_(const char *, const char *, int *, double *, int *, double *, double * work, int * lwork, int * info) {
    if (work && lwork && *lwork == -1) {
        work[0] = 1.0;
    }
    return lapack_error(info);
}

extern "C" int sgesvd_(const char *, const char *, int *, int *, float *, int *, float *, float *, int *, float *, int *, float * work, int * lwork, int * info) {
    if (work && lwork && *lwork == -1) {
        work[0] = 1.0f;
    }
    return lapack_error(info);
}

extern "C" int dgesvd_(const char *, const char *, int *, int *, double *, int *, double *, double *, int *, double *, int *, double * work, int * lwork, int * info) {
    if (work && lwork && *lwork == -1) {
        work[0] = 1.0;
    }
    return lapack_error(info);
}

extern "C" void sgetrf_(int *, int *, float *, int *, int *, int * info) {
    lapack_error(info);
}

extern "C" void sgetri_(int *, float *, int *, int *, float *, int *, int * info) {
    lapack_error(info);
}

extern "C" void dgetrf_(int *, int *, double *, int *, int *, int * info) {
    lapack_error(info);
}

extern "C" void dgetri_(int *, double *, int *, int *, double *, int *, int * info) {
    lapack_error(info);
}

extern "C" int sgelsd_(int *, int *, int *, float *, int *, float *, int *, float *, float *, int *, float * work,
        int * lwork, int *, int * info) {
    if (work && lwork && *lwork == -1) {
        work[0] = 1.0f;
    }
    return lapack_error(info);
}
