from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import mne

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from mne.viz import plot_events
from mne.datasets import fetch_fsaverage
from mne.minimum_norm import make_inverse_operator, apply_inverse

from mne import (
                    read_epochs, 
                    concatenate_epochs,
                    open_report,
                    compute_covariance,
                    make_forward_solution
                )

from mne.decoding import (
                            GeneralizingEstimator,
                            LinearModel,
                            Scaler,
                            Vectorizer,
                            cross_val_multiscore,
                            get_coef,
                            )


def split_epochs(subject, saving_dir):

    ## check paths
    ep_fname = saving_dir / "epochs" / f"{subject}-epo.fif"
    re_fname1 = saving_dir / "reports" / f"{subject}-report.h5" 
    re_fname2 = saving_dir / "reports" / f"{subject}-report.html"
    overwrite = True
    
    if re_fname2.exists():
        return None

    ## read and modify epochs/report
    epochs = read_epochs(ep_fname, preload=True)
    report = open_report(re_fname1)
    sfreq = epochs.info["sfreq"]
    epochs.pick(picks="eeg")
    epochs.drop_bad(reject=dict(eeg=100e-6)) # maybe totally remove this
    even_ids = epochs.event_id
    even_ids.pop("New Segment/", None)

    fig_events, ax = plt.subplots(1, 1, figsize=(10, 4), layout="tight")
    plot_events(epochs.events, sfreq=sfreq, event_id=even_ids, axes=ax, show=False)
    ax.get_legend().remove()
    ax.spines[["right", "top"]].set_visible(False)
    fig_drop = epochs.plot_drop_log(show=False)

    report.add_figure(fig=fig_events, title="Events", image_format="PNG")
    report.add_figure(fig=fig_drop, title="Drop log", image_format="PNG")

    ## balance trials across the 4 classes in training
    rnd_ids = [key for key in even_ids if key.endswith("rndm")]
    eps_list = [epochs[rnd_id] for rnd_id in rnd_ids]
    mne.epochs.equalize_epoch_counts(eps_list, method="mintime")
    epochs_rnd = concatenate_epochs(eps_list)

    ord_ids = [key for key in even_ids if key.endswith("or")]
    epochs_ord = epochs[ord_ids]

    report.add_epochs(epochs_rnd, title="Random trials info", psd=False, projs=False)
    report.add_epochs(epochs_ord, title="Ordered trials info", psd=False, projs=False)

    ## compute covariance matrix from rnd epochs
    cov = compute_covariance(epochs_rnd, tmax=0.0)

    ## define epochs
    ids = range(1, 5) 
    epochs_rnd_std = epochs_rnd[[f"f{i}_std_rndm" for i in ids]]
    epochs_rnd_tin = epochs_rnd[[f"f{i}_tin_rndm" for i in ids]]

    epochs_ord_std = epochs_ord[[f"f{i}_std_or" for i in ids]]
    epochs_ord_tin = epochs_ord[[f"f{i}_tin_or" for i in ids]]

    report.save(saving_dir / "reports" / f"{subject}-report.html",
                overwrite=overwrite, open_browser=False)

    del epochs_ord, epochs_rnd

    return epochs_rnd_std, epochs_rnd_tin, epochs_ord_std, epochs_ord_tin




def decode(subject, saving_dir, epochs_rnd_std, epochs_rnd_tin, epochs_ord_std, epochs_ord_tin):
    
    ###### train clf on random trials
    n_splits = 5
    scores_dir = saving_dir / "scores"
    coeffs_dir = saving_dir / "coeffs"
    stcs_dir = saving_dir / "stcs"
    [sel_dir.mkdir(exist_ok=True) for sel_dir in [scores_dir, coeffs_dir, stcs_dir]]

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

        ## train post -> test post with fit (to extract weights)
        gen.fit(X_post_rnd, y)
        coef_filt = get_coef(gen, "filters_", inverse_transform=False)
        coef_patt = get_coef(gen, "patterns_", inverse_transform=True)[0] # (n_chs, n_class, n_time)

        np.save(coeffs_dir / f"{subject}_rnd_params_{label}.npy", coef_filt)
        np.save(coeffs_dir / f"{subject}_rnd_patterns_{label}.npy", coef_patt)

        ## train post -> test post with cv
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        scores_post_post = cross_val_multiscore(gen, X_post_rnd, y, cv=cv, n_jobs=1)

        ## train post -> test pre
        scores_post_pre = []
        for train_idx, test_idx in cv.split(X_post_rnd, y):
            gen.fit(X_post_rnd[train_idx], y[train_idx]) # train on post
            score = gen.score(X_pre_rnd[test_idx], y[test_idx]) # test on pre
            scores_post_pre.append(score)
        scores_post_pre = np.array(scores_post_pre)

        ## save scores and coeffs 
        np.save(scores_dir / f"{subject}_rnd_post2post_{label}.npy", scores_post_post)
        np.save(scores_dir / f"{subject}_rnd_post2pre_{label}.npy", scores_post_pre)

        ## source space decoding for random post
        stcs = run_source_analysis(coef_patt, epochs_rnd)
        [stc.save(stcs_dir / f"{subject}_rnd_class_{label}_{stc_idx + 1}") for stc_idx, stc in enumerate(stcs)]

        ## test on ordered tones
        X_ord = epochs_ord.get_data()
        y_ord = epochs_ord.events[:, 2]

        times_ord = epochs_ord.times
        post_mask_ord = epochs_ord.times >= 0
        pre_mask_ord  = epochs_ord.times < 0

        X_ord_post = X_ord[:, :, post_mask_ord]
        X_ord_pre  = X_ord[:, :, pre_mask_ord]

        gen.fit(X_post_rnd, y) # train again on random

        ## scores and coeffs
        if label == "standard":
            mapping_ord_2_rnd = dict(zip(range(1, 5), range(5, 9)))
        if label == "tinnitus":
            mapping_ord_2_rnd = dict(zip(range(11, 15), range(15, 19)))
        
        y_ord_mapped = np.array([mapping_ord_2_rnd[val] for val in y_ord])

        score_ord_post = gen.score(X_ord_post, y_ord_mapped)
        score_ord_pre = gen.score(X_ord_pre, y_ord_mapped)

        coef_filt_ord = get_coef(gen, "filters_", inverse_transform=False) # (n_chs, n_class, n_time)
        coef_patt_ord = get_coef(gen, "patterns_", inverse_transform=True)[0] # (n_chs, n_class, n_time) # check this later

        ## save in numpy array  
        np.save(scores_dir / f"{subject}_ord_post2post_{label}.npy", score_ord_post)
        np.save(scores_dir / f"{subject}_ord_post2pre_{label}.npy", score_ord_pre)

        np.save(coeffs_dir / f"{subject}_ord_params_{label}.npy", coef_filt_ord)
        np.save(coeffs_dir / f"{subject}_ord_patterns_{label}.npy", coef_patt_ord)


def run_source_analysis(coef_patt, epochs):

    epochs.set_eeg_reference("average", projection=True)
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
    stcs = []
    for evoked in evokeds:
        stcs.append(
                    apply_inverse(
                            evoked, 
                            inv,
                            lambda2=1.0 / 9.0,
                            method="dSPM",
                            pick_ori="normal"
                            )
                    )
    
    del fwd, inv
    return stcs


if __name__ == "__main__":

    saving_dir = Path("/Volumes/Extreme_SSD/payam_data/Tinreg")
    eps_dir = saving_dir / "epochs"
    subjects = [fname.stem[:4] for fname in sorted(eps_dir.iterdir()) if not fname.stem.startswith(".")]
    for subject in subjects[:2]:
        epochs_list = split_epochs(subject, saving_dir)
        decode(subject, saving_dir, *epochs_list)