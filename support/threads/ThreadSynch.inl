namespace support {

inline void throwError(bool is_fail, int errno) {
  if (is_fail) {
    throw Exception(ErrorCode(errno, getErrorCategory<StdErrCategory>()));
  }
}

/*** Lock ***/
inline Lock::Lock() {
  int errno = pthread_mutex_init(&m_mtx, NULL);
  throwError(errno != 0, errno);
}

inline Lock::~Lock() {
  int errno = pthread_mutex_destroy(&m_mtx);
  throwError(errno != 0, errno);
}

//acquire()
inline void Lock::acquire() {
  pthread_mutex_lock(&m_mtx);
}

//release()
inline void Lock::release() {
  pthread_mutex_unlock(&m_mtx);
}

/*** Barrier ***/
inline Barrier::Barrier(int num_threads) {
  int errno = pthread_barrier_init(&m_bar, NULL, num_threads);
  throwError(errno != 0, errno);
}

inline Barrier::~Barrier() {
  int errno = pthread_barrier_destroy(&m_bar);
  throwError(errno != 0, errno);
}

//wait()
inline void Barrier::wait() {
  int errno = pthread_barrier_wait(&m_bar);
  throwError((errno != 0 && errno != PTHREAD_BARRIER_SERIAL_THREAD),
             errno);
}

/*** Semaphore ***/
inline Semaphore::Semaphore(int value) {
  int errno = sem_init(&m_sem, NULL, value);
  throwError(errno != 0, errno);
}

inline Semaphore::~Semaphore() {
  int errno = sem_destroy(&m_sem);
  throwError(errno != 0, errno);
}

//wait()
inline void Semaphore::wait() {
  int errno = sem_wait(&m_sem);
  throwError(errno != 0, errno);
}

//post()
inline void Semaphore::post() {
  int errno = sem_post(&m_sem);
  throwError(errno != 0, errno);
}

//getValue()
inline void Semaphore::getValue(int* p_val) {
  int errno = sem_getvalue(&m_sem, p_val);
  throwError(errno != 0, errno);
}

} /* namespace support */
