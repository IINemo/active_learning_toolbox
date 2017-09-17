import logging, numpy as np, scipy.stats
from annoy import AnnoyIndex
from sklearn.decomposition import TruncatedSVD
from libact.base.dataset import ensure_sklearn_compat


logger = logging.getLogger('actleto')


class ADWeS(object):
    def __init__(self,
                 dataset,
                 basic_strategy,
                 svd_components=300,
                 index_trees=10,
                 get_nearest_n=10,
                 get_most_uncertain_n=0,
                 exp_rel_power=0.8,
                 exp_rel_rate=1.0,
                 uncertainty_factor=0.5,
                 us_method='lc',
                 plot_each=20):
        self.dataset = dataset
        self.basic_strategy = basic_strategy

        self.get_nearest_n = get_nearest_n
        self.get_most_uncertain_n = get_most_uncertain_n
        self.exp_rel_power = exp_rel_power
        self.exp_rel_rate = exp_rel_rate
        self.uncertainty_factor = uncertainty_factor
        self.us_method = us_method
        self.plot_each = plot_each

        self.index = AnnoyIndex(svd_components)
        all_features = ensure_sklearn_compat(zip(*dataset.data)[0])
        self.data = TruncatedSVD(n_components=svd_components).fit_transform(all_features)
        for i, item in enumerate(self.data):
            self.index.add_item(i, item)
        self.index.build(index_trees)

        self.labeled_ids = set() # will be updated in make_query before all job
        
        # calculate mean and maximum distances
        self.explore_relevance = []
        self.explore_relevance_max = 0
        for i in range(self.data.shape[0]):
            cur_dist = self.index.get_nns_by_item(i,
                                                  self.get_nearest_n,
                                                  include_distances=True)[1]
            if len(cur_dist) > 0:
                cur_mean = np.mean(cur_dist)
                cur_max_dist = np.max(cur_dist)
                if cur_max_dist > self.explore_relevance_max:
                    self.explore_relevance_max = cur_max_dist
            else:
                cur_mean = np.nan
            self.explore_relevance.append(cur_mean)
        self.explore_relevance = np.array(self.explore_relevance)

        # fill na
        samples_without_neighbors = np.isnan(self.explore_relevance)
        self.explore_relevance[samples_without_neighbors] = self.explore_relevance_max

        # normalize
        logger.debug('init dist %s' % str(scipy.stats.describe(self.explore_relevance)))
        self.explore_relevance = ((self.explore_relevance - self.explore_relevance.min()) /
                                  (self.explore_relevance.max() - self.explore_relevance.min()))
        self.explore_relevance = (1 - self.explore_relevance) ** self.exp_rel_power

        self.iter_i = 0

    def make_query(self, return_score=False):
        self._update_exp_rel()

        self.model.train(self.dataset)

        unlabeled_entry_ids, X_pool = list(zip(*self.dataset.get_unlabeled_entries()))
        unlabeled_entry_ids = np.asarray(unlabeled_entry_ids)
        X_pool = ensure_sklearn_compat(X_pool)

        _, ids_with_scores = self.base_strategy.make_query(return_score=True)
        unlabeled_entry_ids, base_score = zip(*ids_with_scores)

        # normalize: we dont care about absolute values, only relative to rank samples
        #base_score = base_score - base_score.mean()
        #base_score /= base_score.std()
        base_score = base_score - base_score.min()
        base_score /= base_score.max()

        if self.get_most_uncertain_n > 0:
            most_base_relevant_indices = np.argpartition(-base_score, self.get_most_uncertain_n)[:self.get_most_uncertain_n]
        else:
            most_base_relevant_indices = list(range(len(base_score)))
        most_base_relevant_ids = unlabeled_entry_ids[most_base_relevant_indices]
        most_base_relevant_score = base_score[most_base_relevant_indices]
        logger.debug('most base relevant score %s' % str(scipy.stats.describe(most_base_relevant_score)))
        most_base_relevant_exp_rel = self.explore_relevance[most_base_relevant_ids]
        
        # normalize: we dont care about absolute values, only relative to rank samples
        #most_uncertain_exp_rel = most_uncertain_exp_rel - most_uncertain_exp_rel.mean()
        #most_uncertain_exp_rel /= most_uncertain_exp_rel.std()
        most_base_relevant_exp_rel = most_base_relevant_exp_rel - most_base_relevant_exp_rel.min()
        most_base_relevant_exp_rel /= most_base_relevant_exp_rel.max()
        logger.debug('most exp rel %s' % str(scipy.stats.describe(most_base_relevant_exp_rel)))

        # f-beta
        result_score = ((1 + self.uncertainty_factor ** 2) * most_base_relevant_score * most_base_relevant_exp_rel /
                        ((self.uncertainty_factor ** 2) * most_base_relevant_score + most_base_relevant_exp_rel))
        #result_score = (self.uncertainty_factor * most_uncertain_uncert_score
        #                + (1 - self.uncertainty_factor) * most_uncertain_exp_rel)
        result_score[np.isnan(result_score)] = 0.0
        logger.debug('most res %s' % str(scipy.stats.describe(result_score)))

#         if self.iter_i % self.plot_each == 0:
#             import matplotlib.pyplot as plt
#             fig, ax = plt.subplots()
#             fig.set_size_inches((9, 6))
#             ax.hist(most_base_relevant_score, label='uncert')
#             ax.hist(most_base_relevant_exp_rel, label='exp_rel')
#             ax.hist(result_score, label='res')
#             fig.savefig('./debug/%05d_hist.png' % self.iter_i)
#             plt.close(fig)
# 
#             _, ax = plot_samples(np.array([most_base_relevant_score,
#                                            most_base_relevant_exp_rel]).T,
#                                  result_score,
#                                  with_kde=False,
#                                  filename='./debug/%05d_scores.png' % self.iter_i,
#                                  do_not_display=True)
#             ax.set_xlabel('uncert')
#             ax.set_ylabel('exp_rel')

        best_i = np.argmax(result_score)
        best_id = most_base_relevant_ids[best_i]
        logger.debug('best %r %r %r %r' % (best_i,
                                           result_score[best_i],
                                           most_base_relevant_score[best_i],
                                           most_base_relevant_exp_rel[best_i]))
        if return_score:
            return best_id, \
                   list(zip(most_base_relevant_ids, result_score))
        else:
            return best_id

    def _update_exp_rel(self):
        data = self.dataset.data
        newly_labeled_ids = { i for i in range(len(data))
                             if not data[i][1] is None
                             and not i in self.labeled_ids }
        self.labeled_ids.update(newly_labeled_ids)
        for ex_id in newly_labeled_ids:
            neighbor_ids, neighbor_dist = self.index.get_nns_by_item(ex_id,
                                                                     self.get_nearest_n,
                                                                     include_distances=True)
            neighbor_dist = np.asarray(neighbor_dist, dtype='float')
            neighbor_discount_factor = (1 - neighbor_dist / self.explore_relevance_max) ** self.exp_rel_power
            neighbor_discount_factor= 1 - self.exp_rel_rate * neighbor_discount_factor
            #logger.debug('dist: %s' % neighbor_dist)
            #logger.debug('factor: %s' % neighbor_discount_factor)
            assert np.count_nonzero(np.isnan(neighbor_discount_factor)) == 0
            self.explore_relevance[neighbor_ids] *= neighbor_discount_factor

        self.iter_i += 1
#         if self.iter_i % self.plot_each == 0:
#             plot_samples(self.data,
#                          self.explore_relevance,
#                          kind='svd',
#                          with_kde=False,
#                          filename='./debug/%05d.png' % self.iter_i,
#                          do_not_display=True)
#             