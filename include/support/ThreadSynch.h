#ifndef SUPPORT_THREADS_SYNCH_H_
#define SUPPORT_THREADS_SYNCH_H_

#include <pthread.h>
#include <semaphore.h>

#include "support/Exception.h"
#include "support/StdErrCategory.h"

/*! \file ThreadSynch.h
 *  \brief Thread synchronization classes. 
 *
 *  ToDo:
 *  Perhaps not a good OO design, should use RIIT.
 */

namespace support {

/*! \class Mutex
 *  A support class for lock, this implementation of locks is expected to be
 *  exception safe, that is, if a thread throws an exception, acquired mutex
 *  will be unlocked.
 */
class Mutex {
 public:
  /*** ctor/dtor ***/
  Mutex();
  ~Mutex();
  /*** gettor/settor ***/
  pthread_mutex_t& get();
 private:
  pthread_mutex_t m_mtx;
};


/*! \class Lock
 */
class Lock {
 public:
  /*** ctor/dtor ***/
  /* Lock()
   * Acquires mutex
   */
  Lock(Mutex& m);

  /* ~Lock()
   * Release mutex
   */
  ~Lock();

 private:
  pthread_mutex_t* m_pMtx;

  //private copy and assignment ctor
  Lock(const Lock& other);
  Lock& operator=(const Lock& other);
};

/*! \class Barrier
 */
class Barrier {
 public:
  /*** ctor/dtor ***/
  Barrier(int num_threads);
  
  ~Barrier();

  /*** methods ***/
  void wait();

 private:
  pthread_barrier_t m_bar;

  //private copy and assignment ctor
  Barrier(const Barrier& other);
  Barrier& operator=(const Barrier& other);
};

/*! \class Semaphore
 */
class Semaphore {
 public:
  /*** ctor/dtor ***/
  Semaphore(int value);
  
  ~Semaphore();

  /*** methods ***/
  void wait();

  void post();

  void getValue(int* p_val);

 private:
  sem_t m_sem;

  //private copy and assignment ctor
  Semaphore(const Semaphore& other);
  Semaphore& operator=(const Semaphore& other);
};

#if 0
/*! \class CondVar
 */
class CondVar {
 public:
  /*** ctor/dtor ***/
  CondVar();
  
  ~Condvar();

  /*** methods ***/
  void wait(Lock& l);

  void signal();

  void broadcast();

 private:
  pthread_mutex_t m_mtx;
  pthread_cond_t m_cond;

  //private copy and assignment ctor
  CondVar(const CondVar& other);
  CondVar& operator=(const CondVar& other);
};
#endif

} /* namespace support */

#include "../lib/support/ThreadSynch.inl"

#endif /* SUPPORT_THREADS_SYNCH_H_ */
