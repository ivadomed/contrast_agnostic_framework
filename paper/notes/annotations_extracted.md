# Review feedback extracted from `PALETTE_draft_20260802.pdf`

72 annotations.

- [ ] **p1** (Text) — paulh
      > Partition-based
      **That's exactly 8 pages (but with some missing stuff). That's not really good. I mean the limit is 8 pages, but after a first review we might be asked to add stuff, so having 7 or a bit more than 7 pages would be better**
- [ ] **p1** (Text) — paulh
      > PALETTE: Partition-based
      **we'll change this title, to something more like "PALETTE-Aug: A simple texture preserving augmentation for medical imaging domain generalization**
- [ ] **p1** (Text) — paulh
      **not sure this abstract is really good enough and structured correctly tbh. maybe not explicitly, but structure it (INTRODUCTION, DATA, METHOD, RESULTS, DISCUSSION)**
- [ ] **p1 L40-42** (Text) — paulh
      > another: than
      **i wouldn't say they are routinely deployed on new modalities...**
- [ ] **p1 L41-43** (Text) — paulh
      > sequence, Because
      **this would require a citation. also, for the introduciton maybe find some inspiration in the synthseg paper and the auglab paper (one sequence to segment them all)**
- [ ] **p1 L43-45** (Text) — paulh
      > rather and
      **it is an important modality, that's for sure, but we don't have any result on it, not sure we should mention it. your call**
- [ ] **p1 L47-49** (Text) — paulh
      > images Collecting
      **citation please**
- [ ] **p1 L51-53** (Text) — paulh
      > and The
      **Good argument. But needs a citation**
- [ ] **p1 L55-57** (Text) — paulh
      > — is and
      **citation?**
- [ ] **p1 L17-19** (Text) — paulh
      > Transform single-source
      **i wouldn't say PALETTE is label free, it can be, and more important it doesn't require dense labels (unlike synthseg), but only the usual sparse segmentation labels. palette can work without any labels, but the best results (and everything we present) rely on the segmentation training labels**
- [ ] **p1 L20-22** (Text) — paulh
      > random affine to
      **not sure it's a good idea to introduce a formula in the abstract. (we're not even definiting each term)**
- [ ] **p1 L62-64** (Text) — paulh
      > layout. This First,
      **maybe recite synthseg here?**
- [ ] **p1 L63-65** (Text) — paulh
      > label-driven: of every
      **maybe explain dense?**
- [ ] **p1 L70-72** (Text) — paulh
      > and unlabelled Layout
      **wouldn't ideally need a citation... no sure we can find this though. otherwise we should mention we demonstrate this (with the appropriate section link)**
- [ ] **p1 L32-34** (Text) — paulh
      > abdominal PALETTE
      **in your result you should likley mention "healthy brain", tumor and ms. and maybe say that on each task overall we beat all other tested methods significantly (we do only matches some method but that's at a finer grain, if you just mention the tasks then the related result is that we win significantly everywhere)**
- [ ] **p1 L35-37** (Text) — paulh
      > – –
      **you should precise augmentation pipeline designed for cross modality training**
- [ ] **p2 L77-78** (Text) — paulh
      > PALETTE
      **hm misleading. it doesn't require labels, but it can rely on the usual sparse segmentation training, and the results we show rely ont this. label optional? ro something like this?**
- [ ] **p2 L130-131** (Text) — paulh
      > under
      **maybe too strong**
- [ ] **p2 L81-83** (Text) — paulh
      > random to the
      **hm you should at least define the function IF you really want to use it here**
- [ ] **p2 L87-89** (Text) — paulh
      **careful. maybe label-optional? or something like this?**
- [ ] **p2 L140-142** (Text) — paulh
      > a controlled
      **maybe mention this earlier somewhere and look for a citation, we might not be the first ones to prove this. We can always say we re-affirm this**
- [ ] **p2 L90-91** (Text) — paulh
      > the texture
      **intact is too strong.**
- [ ] **p2 L146-148** (Text) — paulh
      > against — a
      **is that task level? and what dataset? why only one dataset? and is it really relevant to give numbers in the contributions?**
- [ ] **p2 L96-98** (Text) — paulh
      > single once
      **they will ask when, where how we tuned it, not sure how to answer. you could just say the training configuraiton is fixed, with only the number of epochs changing between datasets**
- [ ] **p2 L98-100** (Text) — paulh
      > evaluate spanning
      **In the main paper we only present the 4 task level results, not the 8 settings, so maybe only talk about this here? what do you think?**
- [ ] **p2 L99-101** (Text) — paulh
      > spanning AMOS/SLIVER07), brain
      **unclear and maybe not necessary in the introduction, and by the way i think you're missing at least one dataset**
- [ ] **p2 L99-101** (Text) — paulh
      > abdominal (ON-
      **healthy brain, MS brain and tumor brain**
- [ ] **p2 L154-156** (Text) — paulh
      > texture
      **i would say label generative synthesis like synthseg destroys, or softer, don't preserve (or similar)**
- [ ] **p2 L104-106** (Text) — paulh
      > public and
      **unclear**
- [ ] **p2 L104-106** (Text) — paulh
      > implementation [23],
      **you're missing srcsm aren't you?**
- [ ] **p2 L105-107** (Text) — paulh
      > recent com-
      **just a strong competitor is enough**
- [ ] **p2 L159-161** (Text) — paulh
      > matches or and
      **exceeds on every task, and matches or exceed on every settings**
- [ ] **p2 L165** (Text) — paulh
      **this part lack any kind of information/models/augmentation or whatever related to texture in medical imaging**
- [ ] **p2 L167-169** (Text) — paulh
      > targets
      **not a big fan of this abbreviation that is not that useful to be honest**
- [ ] **p2 L169-171** (Text) — paulh
      > target which
      **maybe not useful**
- [ ] **p2 L120-121** (Text) — paulh
      > lesions
      **if you update to report the task level, don't forget to update this**
- [ ] **p2 L122-124** (Text) — paulh
      > mechanism; We
      **not sure about this if we present at the task level. however, this is a proof of humility. maybe just reframe as "some win are only by a thin margin therefore blablabla controlled test"**
- [ ] **p3 L182-184** (Text) — paulh
      > source and
      **we should have a justification for why we have this setting, with a citation**
- [ ] **p3 L235-237** (Text) — paulh
      > The closest which
      **that's srcsm right? maybe you should introduce this name here?**
- [ ] **p3 L238-240** (Text) — paulh
      **wrong, we use it too**
- [ ] **p3 L246-247** (Text) — paulh
      **not sure about this section**
- [ ] **p3 L249-251** (Text) — paulh
      > testing across
      **actually i'm pretty sure auglab (one sequence to segment them all), does pretty much this but with only spine. and actually, i don't see the point of saying that nobody does this (which is not true). and note that for example for MS and CHAOS, we have one training dataset but several testing datasets (including the test set of the training set)**
- [ ] **p3 L198-200** (Text) — paulh
      > structure The
      **citation?**
- [ ] **p3 L250-252** (Text) — paulh
      **be clearer about our 4 tasks (soon to be 6)**
- [ ] **p3 L201-203** (Text) — paulh
      > training-free We
      **not sure i understand this**
- [ ] **p3 L202-204** (Text) — paulh
      > (noEM) variant — and
      **add something like requires dense label maps because it ignores the original anatomical volumes and only relies on the labels to create the artificial volume**
- [ ] **p3 L205-207** (Text) — paulh
      > segmen- — as
      **unclear, what do you mean the labels estimate stuff?**
- [ ] **p3 L206-208** (Text) — paulh
      > dataset to as the
      **the sentence is buchered and hard to understand**
- [ ] **p3 L207-208** (Text) — paulh
      > public code,
      **explicitely say that it is based on paper**
- [ ] **p3 L210-212** (Text) — paulh
      > manufactures
      **is there a first line in the text somewhere?**
- [ ] **p3 L217-219** (Text) — paulh
      > BigAug [37].
      **although it's not optimized for domain generalization (or is it?)**
- [ ] **p3 L268-270** (Text) — paulh
      > at every on one
      **i would appreciate if the figure was actually closer to this part of the manuscript**
- [ ] **p3 L271-272** (Text) — paulh
      > Intensity
      **In the figure 1. is the input image and 2. is the k-mean intensity remap,... your step should match the figure or the figure should match your steps**
- [ ] **p3 L230-231** (Text) — paulh
      > architecture-agnostic but
      **don't write this, it's not the only one, and GIN as annoying differences making it hard to compare**
- [ ] **p4 L338-340** (Text) — paulh
      > segmentation – whatever
      **very important, worth stating later in the paper too**
- [ ] **p4 L294-295** (Text) — paulh
      > For
      **In the figure this is the very last step which produces the last outputs**
- [ ] **p4 L313-314** (Text) — paulh
      > Steps
      **Actually, in the ablation, this happens before the voronoi step, so this should be put before it both in the prose and in the figure. also, the name is misleading. you could just say that optionnally we use the labels to overwrite the voxels belonging to each label is like overwritten or reassgined to a new region but it herits the k-means and subparcellation of the previous steps (you should maybe directly look into the code to fully understand)**
- [ ] **p5 L410-412** (Text) — paulh
      **add something like: as it is usually done in medical imaging, and to provide robust statistical analysis**
- [ ] **p5 L413-414** (Text) — paulh
      > We
      **we'll have to provide more details about the datasets (number of subjects, protocols etc etc in the supplementary), also the fact  that on harmony labels were made with silver standard labels produced by freesurfer (find the silver standard reference somewhere) on t1w and then registarted on each other modality (a patient always has T1w and other modalities)**
- [ ] **p5 L392** (Text) — paulh
      **I don't see the "we prove that palette and auglab have texture by using NGF on their volumes" analysis. i think it could be super useful and help explain why we're only winning by a thin margin on brats against auglab (according to the missing analysis, auglab preserves texture too!)**
- [ ] **p5 L422-424** (Text) — paulh
      > CHAOS together
      **only on the held out test set of course**
- [ ] **p5 L424-426** (Text) — paulh
      > cross-contrast):
      **healthy brain segmentation, i would even say 31-region helthy healthy brain segmentation**
- [ ] **p6 L427-429** (Text) — paulh
      > complementary structural Reference
      **only on the test set, and  held our contrasts but also the training contrasts (but for all contrasts, only test set!)**
- [ ] **p6 L437-438** (Text) — paulh
      > held-out contrasts
      **same, only test set and also tested on the training contrast**
- [ ] **p6 L440-442** (Text) — paulh
      > the nnU-Net [5],
      **explain EM and noEM somewhere please (EM = expectation maximization). explain it before you use EM for the first time please**
- [ ] **p6 L478-480** (Text) — paulh
      > which resamples is competitive
      **dense label map only, with no anatomical scan information (or something like this)**
- [ ] **p6 L484-485** (Text) — paulh
      > The
      **not sure if you're only talking about table 1 or if you're also thinking about setting level results in the supplementary. if that's the case, explicitly mention the supplementary and make it clear if you're talking about it.**
- [ ] **p6 L498-500** (Text) — paulh
      > non-significant the
      **again, not the case in the table above, only in the supplementary materials. it's still more fair to address it directly rather thank hide it, but make it clear you're reference setting level results presented in the supplementary**
- [ ] **p6 L501-503** (Text) — paulh
      > are PALETTE
      **careful, not always true, maybe don't say that, at least not as a generality**
- [ ] **p7** (Text) — paulh
      > +7.70
      **here too, i prefer tissue interface and no tissue interface**
- [ ] **p7** (Text) — paulh
      > The
      **In this figure i read texture defined and boundary defined. we've never proved this, instead you should probably write tissue interface and no tissue interface (or something like this)**
- [ ] **p7 L524** (Text) — paulh
      **small issue, did we isolate the blur thing? or do we bring back the affine remap and the blur at the same time? (check the code)**
