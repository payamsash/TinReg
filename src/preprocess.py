from pathlib import Path
import datetime
import matplotlib.pyplot as plt

from mne_icalabel import label_components
from mne.io import read_raw_brainvision
from mne.channels import make_standard_montage
from mne.viz import plot_projs_joint
from mne import (
                events_from_annotations,
                Epochs,
                Report,
                )
from mne.preprocessing import (
                                ICA,
                                create_eog_epochs,
                                compute_proj_eog,
                                find_bad_channels_lof
                                )

def preprocess(subject, main_dir, saving_dir, use_ssp):

    paradigm = "regularity"
    if use_ssp:
        ch_types = {
                    "O1": "eog",
                    "O2": "eog",
                    "PO7": "eog",
                    "PO8": "eog",
                    "Pulse": "ecg",
                    "Resp": "ecg",
                    "Audio": "stim"
                }
        eog_chs_1 = ["PO7", "PO8"]
        eog_chs_2 = ["O1", "O2"]
    else:
        ch_types = {
                    "Pulse": "ecg",
                    "Resp": "ecg",
                    "Audio": "stim"
                }

    manual_data_scroll = False
    montage = make_standard_montage("easycap-M1")
    shift_in_ms = 0 # need to check later
    sfreq_1 = 1000
    sfreq_2 = 100
    (tmin, tmax) = (-0.4, 0.5)
    show = False

    ## reading and preprocessing the files
    ep_dir = saving_dir / "epochs"
    re_dir = saving_dir / "reports"
    [sel_dir.mkdir(exist_ok=True) for sel_dir in [ep_dir, re_dir]]
    ep_fname = ep_dir / f"{subject}-epo.fif"
    re_fname = re_dir / f"{subject}-report.html" 
    
    if ep_fname.exists():
        return None

    fname = main_dir / f"{subject}_{paradigm}.vhdr"
    raw = read_raw_brainvision(fname, preload=True)
    raw.set_channel_types(ch_types)
    raw.pick(["eeg", "eog", "ecg", "stim"])
    raw.set_montage(montage=montage, match_case=False, on_missing="warn")
    events, events_dict = events_from_annotations(raw)

    if raw.info["sfreq"] > 1000.0:
        raw, events = raw.resample(sfreq_1, stim_picks=None, events=events)

    noisy_chs, lof_scores = find_bad_channels_lof(raw, threshold=3, return_scores=True)
    raw.info["bads"] = noisy_chs

    if manual_data_scroll:
        raw.annotations.append(onset=0, duration=0, description="bad_segment")
        raw.plot(duration=20.0, n_channels=80, picks="eeg", scalings=dict(eeg=40e-6), block=True)
    
    if len(raw.info["bads"]):
        raw.interpolate_bads()
    
    ## filtering
    raw.filter(0.1, 30)
    raw.set_eeg_reference("average", projection=False)
    
    
    if use_ssp:

        ## vertical eye movement
        ev_eog = create_eog_epochs(raw, ch_name=eog_chs_1).average(picks="all")
        ev_eog.apply_baseline((None, None))
        veog_projs, _ = compute_proj_eog(raw, n_eeg=1, reject=None) # so only blink
        raw.add_proj(veog_projs)
        raw.apply_proj()

        ## horizontal eye movement
        try:
            ica = ICA(n_components=0.97, max_iter=800, method='infomax', fit_params=dict(extended=True))        
        except:
            ica = ICA(n_components=5, max_iter=800, method='infomax', fit_params=dict(extended=True)) 
    
        ica.fit(raw)
        eog_indices, eog_scores = ica.find_bads_eog(raw, ch_name=eog_chs_2, threshold=1.2)
        eog_indices_fil = [x for x in eog_indices if x <= 10]
        heog_idxs = [eog_idx for eog_idx in eog_indices_fil if eog_scores[0][eog_idx] * eog_scores[1][eog_idx] < 0]
        fig_scores = ica.plot_scores(scores=eog_scores, exclude=eog_indices_fil, show=show)

        if len(heog_idxs) > 0:
            eog_sac_components = ica.plot_properties(
                                                        raw,
                                                        picks=heog_idxs,
                                                        show=show,
                                                        )
            ica.apply(raw, exclude=heog_idxs)

    else:
        try:
            ica = ICA(n_components=0.97, max_iter=800, method='infomax', fit_params=dict(extended=True))        
        except:
            ica = ICA(n_components=5, max_iter=800, method='infomax', fit_params=dict(extended=True)) 
    
        ica.fit(raw)
        ic_dict = label_components(raw, ica, method="iclabel")
        ic_labels = ic_dict["labels"]
        ic_probs = ic_dict["y_pred_proba"]
        eog_indices = [idx for idx, label in enumerate(ic_labels) \
                        if label == "eye blink" and ic_probs[idx] > 0.70]
        eog_indices_fil = [x for x in eog_indices if x <= 10]

        if len(eog_indices) > 0:
            eog_components = ica.plot_properties(raw,
                                                picks=eog_indices_fil,
                                                show=show,
                                                )
            
        ica.apply(raw, exclude=eog_indices_fil)
    
    ## create report
    report = Report(title=f"report_subject_{subject}")
    report.add_raw(raw=raw, title="Recording Info", butterfly=False, psd=True)

    if use_ssp:
        fig_ev_eog, ax = plt.subplots(1, 1, figsize=(7.5, 3))
        ev_eog.plot(picks="PO7", time_unit="ms", titles="", axes=ax, show=show)
        ax.set_title("Vertical EOG")
        ax.spines[["right", "top"]].set_visible(False)
        ax.lines[0].set_linewidth(2)
        ax.lines[0].set_color("magenta")
        ev_eog.apply_baseline((None, None))

        fig_eog = ev_eog.plot_joint(picks="eeg", ts_args={"time_unit": "ms"}, show=show)
        fig_proj = plot_projs_joint(veog_projs, ev_eog, picks_trace="Fp1", show=show)

        for fig, title in zip([fig_ev_eog, fig_eog, fig_proj, fig_scores], ["Vertical EOG", "EOG", "EOG Projections", "Scores"]):
            report.add_figure(fig=fig, title=title, image_format="PNG")
        if len(heog_idxs) > 0:
            report.add_figure(fig=eog_sac_components, title="EOG Saccade Components", image_format="PNG")
    else:
        if len(eog_indices) > 0:
            report.add_figure(fig=eog_components, title="EOG Components", image_format="PNG")

    ## epoching and saving
    events[:, 0] = events[:, 0] + shift_in_ms
    trigger_dict = {
        "f1_std_or": 1,
        "f2_std_or": 2,
        "f3_std_or": 3,
        "f4_std_or": 4,
        "f1_std_rndm": 5,
        "f2_std_rndm": 6,
        "f3_std_rndm": 7,
        "f4_std_rndm": 8,
        "f1_tin_or": 11,
        "f2_tin_or": 12,
        "f3_tin_or": 13,
        "f4_tin_or": 14,
        "f1_tin_rndm": 15,
        "f2_tin_rndm": 16,
        "f3_tin_rndm": 17,
        "f4_tin_rndm": 18
    }
    epochs = Epochs(
                        raw,
                        events,
                        trigger_dict,
                        tmin=tmin,
                        tmax=tmax,
                        baseline=None, # no baselining
                        preload=True,
                        )
    epochs.resample(sfreq_2)
    evoked = epochs.average()
    report.add_evokeds(evoked)

    epochs.save(ep_fname, overwrite=True)
    report.save(re_dir / f"{subject}-report.h5", open_browser=False, overwrite=True)
    del raw

if __name__ == "__main__":
    
    main_dir = Path("/Volumes/G_USZ_ORL$/Research/ANTINOMICS/data/eeg")
    saving_dir = Path("/Volumes/Extreme_SSD/payam_data/Tinreg")
    cutoff_date = datetime.datetime(2025, 6, 3) # 3rd June
    
    paradigm = "regularity"
    subjects = []
    ssp_usage = []
    for fname in sorted(main_dir.iterdir(), key=lambda f: f.stat().st_mtime):
        if str(fname).endswith(f"{paradigm}.vhdr"):
            mtime = datetime.datetime.fromtimestamp(fname.stat().st_mtime)
            if mtime < cutoff_date:
                ssp_usage.append(True)
            else:
                ssp_usage.append(False)
            subjects.append(fname.stem.split("_")[0])

    subjects_to_remove = ["vuio", "nrjq"]

    for subject, use_ssp in zip(subjects, ssp_usage):
        if subject in subjects_to_remove:
            continue
        preprocess(subject, main_dir, saving_dir, use_ssp)
