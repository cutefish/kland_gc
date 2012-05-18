namespace support {

inline void throwError(bool is_fail, int error) {
  if (is_fail) {
    throw Exception(ErrorCode(error, getErrorCategory<StdErrCategory>()));
  }
}

/*** Lock ***/
inline Lock::Lock() {
  int error = pthread_mutex_init(&m_mtx, NULL);
  throwError(error != 0, error);
}

inline Lock::~Lock() {
  int error = pthread_mutex_destroy(&m_mtx);
  throwError(error != 0, error);
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
  int error = pthread_barrier_init(&m_bar, NULL, num_threads);
  throwError(error != 0, error);
}

inline Barrier::~Barrier() {
  int error = pthread_barrier_destroy(&m_bar);
  throwError(error != 0, error);
}

//wait()
inline void Barrier::wait() {
  int error = pthread_barrier_wait(&m_bar);
  throwError((error != 0 && error != PTHREAD_BARRIER_SERIAL_THREAD),
             error);
}

/*** Semaphore ***/
inline Semaphore::Semaphore(int value) {
  int error = sem_init(&m_sem, NULL, value);
  throwError(error != 0, error);
}

inline Semaphore::~Semaphore() {
  int error = sem_destroy(&m_sem);
  throwError(error != 0, error);
}

//wait()
inline void Semaphore::wait() {
  int error = sem_wait(&m_sem);
  throwError(error != 0, error);
}

//post()
inline void Semaphore::post() {
  int error = sem_post(&m_sem);
  throwError(error != 0, error);
}

//getValue()
inline void Semaphore::getValue(int* p_val) {
  int error = sem_getvalue(&m_sem, p_val);
  throwError(error != 0, error);
}

} /* namespace support */
