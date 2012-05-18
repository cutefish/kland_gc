#include "support/Exception.h"
#include "support/StdErrCategory.h"

namespace support {

inline ThreadsAttr::ThreadsAttr() {
  pthread_attr_init(&m_attr);
  pthread_attr_setdetachstate(&m_attr, PTHREAD_CREATE_JOINABLE);
}

inline ThreadsAttr::~ThreadsAttr() {
  pthread_attr_destroy(&m_attr);
}

inline pthread_attr_t* ThreadsAttr::get() {
  return &m_attr;
}

inline void ThreadsAttr::setAffinity() {
  m_isAffinity = true;
}

inline ThreadPool::ThreadPool(int num_threads, ThreadsAttr& attr) 
    : m_numThreads(num_threads), m_wait(false), m_exit(false), m_joined(false),
    m_func(NULL), 
    m_bar(num_threads + 1), m_threads(num_threads),
    m_userArgs(num_threads), m_threadArgs(num_threads)
{

  for (int i = 0; i < num_threads; ++i) {
    m_threadArgs[i].tid = i;
    m_threadArgs[i].pool = this;

    int error = pthread_create(&m_threads[i], attr.get(), &loop, &m_threadArgs[i]);
    throwError(error != 0, error);
  }
}

inline ThreadPool::~ThreadPool() {
  if (!m_joined) {
    join();
  }
}

inline void ThreadPool::set(UserThreadFunc func, std::vector<void*> user_args) {
  m_func = func;
  for (int i = 0; i < m_numThreads; ++i) {
    m_userArgs[i] = user_args[i];
  }
}

inline void ThreadPool::execute() {
  //control thread enters the barrier, which will release all the working
  //threads.
  m_wait = false;
  m_bar.wait();
}

inline void ThreadPool::wait() {
  //control thread enters a barrier, waiting for the other working threads.
  //The working threads will then enter another barrier to wait.
  m_wait = true;
  m_bar.wait();
}

inline void ThreadPool::exit() {
  m_exit = true;
  if (m_wait == true) {
    m_bar.wait();
  }
}

inline std::vector<void*> ThreadPool::join() {
  std::vector<void*> ret;
  for (int i = 0; i < m_numThreads; ++i) {
    ret.push_back(NULL);
    int error = pthread_join(m_threads[i], &ret[i]);
    throwError(error != 0, error);
  }
  m_joined = true;
  return ret;
}

inline void* ThreadPool::loop(void* args) {
  ThreadArgs* thread_args = (ThreadArgs*)args;

  int tid = thread_args->tid;
  ThreadPool* pool = thread_args->pool;

  //enter barrier and wait for start
  pool->m_bar.wait();

  while (!pool->m_exit) {
    if (pool->m_wait) {
      //enter first barrier to release control thread
      pool->m_bar.wait();
      //enter second barrier to wait for control thread
      pool->m_bar.wait();
    }
    else {
      UserThreadFunc func = pool->m_func;
      void* user_args = pool->m_userArgs[tid];
      (*func)(pool, tid, user_args);
    }
  }

  pthread_exit(args);
}


} /* namespace support */
