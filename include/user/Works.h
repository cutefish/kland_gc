#ifndef USER_WORKS_H_
#define USER_WORKS_H_

#include <list>

struct TaskRange {
  int temp_start;
  int temp_end;
  int cont_start;
  int cont_end;
};

/* partition()
 * This partition solves an optimization problem.
 *
 * We need to finish all <num_temp> * <num_cont> calculations with <num_proc>
 * processes while keep the storage footprint of each process minimum. The
 * problem can be formulated as follows.  
 *
 * min n_t * S_t + n_c * S_c, s.t.  
 *
 * N_p * n_t * n_c = N_t * N_c;
 * n_t <= N_t;
 * n_c <= N_c;
 *
 * where n_t and n_c are number of * temp and cont for each process, N_t and N_c
 * are the total number of temp and * cont, N_p is the number of processes, S_t
 * and S_c are the size of storage * footprint for temp and cont.
 *
 * The solution is 
 * opt = sqrt(N_t * N_c * S_c / N_p / S_t);
 * n_t = N_t if opt > * N_t;
 * n_t = opt if otherwise;
 * n_c = N_t * N_c / N_p / n_t;
 */
std::list<TaskRange> partition(int num_proc, 
                               int num_temp, int num_cont, 
                               int temp_npts, int cont_npts);

/* partition()
 * This partition solves the optimization problem from aother aspect
 *
 * There is a relationship between the 
 */
int partition(int num_temp, int num_cont, 
              int temp_npts, int cont_npts,
              size_t capacity, int ratio, 
              std::list<TaskRange>& range_list);

#endif /* USER_PARTITION_H_ */
