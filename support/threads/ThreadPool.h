#ifndef SUPPORT_THREADS_THREADPOOL_H_
#define SUPPORT_THREADS_THREADPOOL_H_

#include <vector>
#include <pthread.h>

#include "ThreadSynch.h"

/*! \file ThreadPool.h
 *  \brief Threads creation, exectuion and termination wrapper.
 */

namespace support {

/*! \callback UserThreadFunc
 *  \brief User defined thread function, will be called in while loop for each
 *  thread
 */
typedef void (*UserThreadFunc)(ThreadPool* pool, int tid, void* args);

/*! \class ThreadsAttr
 *  \brief Wrapper for pthread attributes
 */
class ThreadsAttr {
 public:
  /*** ctor/dtor ***/
  ThreadsAttr();

  ~ThreadsAttr();

  /*** methods ***/
  pthread_attr_t* get();

  void setAffinity();
  
 private:
  pthread_attr_t m_attr;
  bool m_isAffinity;

  //private copy and assignment ctor
  ThreadsAttr(const ThreadsAttr& other);
  ThreadsAttr& operator=(const ThreadsAttr& other);
};

/*! \class ThreadPool
 *  \brief A group of threads, continuesly executing <UserThreadFunc> f in a
 *  while loop. The user functions are specified using set(). Threads can be
 *  synchronized by execute(), wait(). The pool will exit after exit().
 */
class ThreadPool {
 public:
  /*** ctor/dtor ***/
  ThreadPool(int num_threads, ThreadsAttr& attr); //To Do, wrap this attr

  ~ThreadPool();

  /*** methods ***/

  /* set()
   * set <UserThreadFunc> func and args for each thread.
   */
  void set(UserThreadFunc func, std::vector<void*> user_args);

  /* execute()
   * signal execute for the pool, should be called only by control thread.
   */
  void execute();

  /* wait()
   * signal wait for the pool, should be called only by control thread.
   */
  void wait();

  /* exit()
   * signal exit for the pool.
   */
  void exit();

 private:
  struct ThreadArgs {
    int tid;
    ThreadPool* pool;
  };

  /*** member variables ***/
  int m_num_threads;
  volatile bool m_wait;                          //signal waiting
  volatile bool m_exit;                          //signal exit
  UserThreadFunc m_func;
  Barrier m_bar;                        //used by control thread to wait
  std::vector<pthread_t> m_threads;
  std::vector<void*> m_userArgs;
  std::vector<ThreadArgs> m_threadArgs;

  /*** private methods ***/
  static void (*loop)(void*);


  //private copy and assignment ctor
  ThreadPool(const ThreadPool& other);
  ThreadPool& operator=(const ThreadPool& other);
};

} /* namespace support */

#include "ThreadPool.inl"

#endif /* SUPPORT_THREADS_THREADPOOL_H_ */
