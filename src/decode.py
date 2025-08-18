import numpy as np
import matplotlib.pyplot as plt
import mne

from mne.decoding import (
    GeneralizingEstimator,
    LinearModel,
    Scaler,
    SlidingEstimator,
    Vectorizer,
    cross_val_multiscore,
    get_coef,
)

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold

from mne import (
                events_from_annotations,
                Epochs,
                open_report,
                compute_covariance
                )


def split_epochs(subject_id):


    ## read and modify epochs/report
    epochs = mne.read_epochs(f"../sample/epochs/{subject_id}-epo.fif", preload=True)
    report = open_report(fname_report)
    sfreq = epochs.info["sfreq"]
    epochs.pick(picks="eeg")
    epochs.drop_bad(reject=dict(eeg=40e-6))
    even_ids = epochs.event_id
    even_ids.pop("New Segment/", None)

    fig_events, ax = plt.subplots(1, 1, figsize=(10, 4), layout="tight")
    mne.viz.plot_events(epochs.events, sfreq=sfreq, event_id=even_ids, axes=ax)
    ax.get_legend().remove()
    ax.spines[["right", "top"]].set_visible(False)
    fig_drop = epochs.plot_drop_log(show=False)

    report.add_figure(fig=fig_events, title="Events", image_format="PNG")
    report.add_figure(fig=fig_drop, title="Drop log", image_format="PNG")

    ## balance trials across the 4 classes in training
    rnd_ids = [key for key in even_ids if key.endswith("rndm")]
    eps_list = [epochs[rnd_id] for rnd_id in rnd_ids]
    mne.epochs.equalize_epoch_counts(eps_list, method="mintime")
    epochs_rnd = mne.concatenate_epochs(eps_list)

    ord_ids = [key for key in even_ids if key.endswith("or")]
    epochs_ord = epochs[ord_ids]

    report.add_info(epochs_rnd.info, title="Random trials info")
    report.add_info(epochs_ord.info, title="Ordered trials info")

    ## compute covariance matrix from rnd epochs
    cov = compute_covariance(epochs_rnd, tmax=0.0)

    ## define epochs
    ids = range(1, 5) 
    epochs_rnd_std = epochs_rnd[[f"f{i}_std_rndm" for i in ids]]
    epochs_rnd_tin = epochs_rnd[[f"f{i}_tin_rndm" for i in ids]]

    epochs_ord_std = epochs_ord[[f"f{i}_std_or" for i in ids]]
    epochs_ord_tin = epochs_ord[[f"f{i}_tin_or" for i in ids]]

    del epochs_ord, epochs_rnd

    return epochs_rnd_std, epochs_rnd_tin, epochs_ord_std, epochs_ord_tin




def decode(epochs_rnd_std, epochs_rnd_tin, epochs_ord_std, epochs_ord_tin):
    
    ###### train clf on random trials
    n_splits = 5

    labels = ["standard", "tinnitus"]
    for epochs_rnd, epochs_ord, label in zip([epochs_rnd_std, epochs_rnd_tin], [epochs_ord_std, epochs_ord_tin], labels):

        post_mask = epochs_rnd.times >= 0
        pre_mask  = epochs_rnd.times < 0
        X = epochs_rnd.get_data()
        y = epochs_rnd.events[:, 2]

        X_post_rnd = X[:, :, post_mask]
        X_pre_rnd = X[:, :, pre_mask]

        ## define and fit generalization object
        clf = make_pipeline(
                            Scaler(epochs_rnd.info),
                            Vectorizer(),           
                            LinearModel(LinearDiscriminantAnalysis(solver="svd"))
                            )
        gen = GeneralizingEstimator(clf, scoring="accuracy", n_jobs=1, verbose=True)
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

        ## post -> post
        scores_post_post = cross_val_multiscore(gen, X_post_rnd, y, cv=cv, n_jobs=1)

        ## post -> pre
        scores_post_pre = []
        for train_idx, test_idx in cv.split(X_post_rnd, y):
            gen.fit(X_post_rnd[train_idx], y[train_idx]) # train on post
            score = gen.score(X_pre_rnd[test_idx], y[test_idx]) # test on pre
            scores_post_pre.append(score)

        scores_post_pre = np.array(scores_post_pre)

        ## compute coefficients
        coef_filt = get_coef(gen, "filters_", inverse_transform=False) # (n_chs, n_class, n_time)
        coef_patt = get_coef(gen, "patterns_", inverse_transform=True)[0] # (n_chs, n_class, n_time)

        ## save in numpy array  
        np.save("", score ...)

        ## source space decoding
        run_source_analysis(coef_patt, epochs_rnd)

        ## test on ordered tones
        X_ord = epochs_ord.get_data()
        y_ord = epochs_ord.events[:, 2]

        times_ord = epochs_ord.times
        post_mask_ord = epochs_ord.times >= 0
        pre_mask_ord  = epochs_ord < 0

        X_ord_post = X_ord[:, :, post_mask_ord]
        X_ord_pre  = X_ord[:, :, pre_mask_ord]

        gen.fit(X_post, y) # train again on random

        ## scores and coeffs
        score_ord_post = gen.score(X_ord_post, y_ord)
        score_ord_pre = gen.score(X_ord_pre, y_ord)

        coef_filt_ord = get_coef(gen, "filters_", inverse_transform=False) # (n_chs, n_class, n_time)
        coef_patt_ord = get_coef(gen, "patterns_", inverse_transform=True)[0] # (n_chs, n_class, n_time)

        ## save in numpy array  
        np.save("", score ...)


def run_source_analysis(coef_patt, epochs):

    evokeds = []
    for i_cls in range(4):
        evokeds.append(
                        mne.EvokedArray(coef_patt[:, i_cls, :],
                                        epochs.info,
                                        tmin=epochs.times[0])
                        )

    noise_cov = compute_covariance(epochs, tmax=0.0)
    kwargs = {
                "subject": "fsaverage",
                "subjects_dir": None
            }

    fs_dir = fetch_fsaverage()
    trans = fs_dir / "bem" / "fsaverage-trans.fif"
    src = fs_dir / "bem" / "fsaverage-ico-5-src.fif"
    bem = fs_dir / "bem" / "fsaverage-5120-5120-5120-bem-sol.fif"

    fwd = make_forward_solution(
                                epochs.info,
                                trans=trans,
                                src=src,
                                bem=bem,
                                meg=False,
                                eeg=True
                                )
    inv = make_inverse_operator(
                                epochs.info,
                                fwd,
                                noise_cov
                                )
    
    for evoked in evokeds:
        stc = apply_inverse(
                            evoked, 
                            inv,
                            lambda2=1.0 / 9.0,
                            method="dSPM",
                            pick_ori="normal"
                            )
    
    del fwd, inv
    
    stc.save("...")