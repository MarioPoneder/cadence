# Common missteps

These are general lessons for building tasks around Cadence. Frozen
experiment receipts remain the authority for individual runs; this guide is
not a history of benchmark outcomes.

| Misstep | Why it fails | Check or correction |
| --- | --- | --- |
| Treating a normal form as a correct answer | The declared equations can agree while modeling the task poorly. | Measure predictions and executed behavior against observations. |
| Reading free-state energy as surprise | The temporal causal recurrence has zero defect even for a wrong prediction. | Compare a prediction with actual sensor readback; retain the parameter revision that made it. |
| Treating two converged detunings as an accurate learning direction | Positive local curvature does not prove that both solutions remain on the free state's smooth branch. | Check finite-beta sensitivity and actual free loss; reject harmful candidate steps rather than relying on solver convergence alone. |
| Accepting any step along a useful direction | A finite parameter step can overshoot despite a correct local derivative. | Replay candidates from the original boundary, reduce the step if needed and count the extra computation. |
| Calling more replay more data | Reusing the same episodes changes training exposure, not distinct experience. | Vary unique examples, update count and model capacity separately with fixed ports and held-out tasks. |
| Calling action carry sensor feedback | Advancing predicted hidden state after an action does not assimilate its measured consequence. | Declare which actual readings enter the next inference or teaching call; test changed environmental conditions. |
| Treating afterglow as demonstrated short-term memory | Persistent activity may lose the relevant distinction. | Remove the cue, vary the delay and context, and reset activity in controls. |
| Treating protected responses as valuable memory | Constraints preserve an error or a poor policy as faithfully as a useful one. | Check usefulness, later correction and remaining learning capacity together. |
| Counting free memory dimensions as easy learning capacity | A new activity may have only a tiny component outside the protected span. | Measure conditioning, required weight changes and retained behavior during new learning. |
| Calling a goal-detuned prediction a successful plan | A relaxed future can approach the goal even when no action can cause it. | Re-evaluate the action with the goal removed, then execute it and measure the outcome. |
| Treating achievement of a supplied goal as desirable behavior | A model can accurately pursue a target that the user dislikes. | Evaluate the target and the executed result separately; retain user feedback and test learned goal formation explicitly. |
| Treating a causal poset as the memory payload | Dependency order provides provenance, not the content or relevance of an experience. | Specify the retained state and how future decisions can read it. |
| Feeding imagined outcomes back as observations | Predictions can reinforce their own mistakes without new evidence. | Keep actual readings, proposed actions, teaching and imagination distinct. |
| Turning sensor summaries into action commands by name | A brightness measurement is not a pitch command; a detected onset count is not a request for retriggers. | Validate the instrument with known actions and measure its sound or physical readback. |
| Expecting scale to restore discarded information | Many distinct event sequences may have the same compressed summary. | Test encoding collisions and expose the needed distinctions explicitly. |
| Dividing every predicted event grid by its own maximum | A flat positive grid becomes active everywhere after thresholding. | Include constant and silent controls; inspect false and missing actions. |
| Reporting low average error as high task quality | Numerous easy outputs can hide errors in timing, intention or rare decisive events. | Report behavior-specific errors and obtain task-appropriate human assessment. |
| Calling a larger hidden state a global workspace | More capacity does not establish useful coordination or functional specialization. | Compare bounded feedback with matched, disconnected and shuffled controls. |
| Calling ordinary recurrence recursive self-modeling | A copied linear readout can be absorbed into recurrent weights. | Identify information in the summary and show a causal benefit from reading it. |
| Teaching opposite cues to a bias-free odd model without shared context | Opposite inputs force opposite outputs even when the desired behaviors share positive components. | Check representability before training; declare any common context or activation. |
| Inferring learned musical improvement after changing the player | Timbre, thresholds and playback changes can dominate what is heard. | Hold model and instrument fixed in separate comparisons and label the changed component. |
| Interpreting different comments about identical audio as progress | Playback context and expectations can affect listening reports. | Hash clips, disclose duplicates and preserve verbatim feedback separately from metrics. |
| Learning a few phrases and claiming original composition | Recall, interpolation, held-out transfer and intentional invention are different tasks. | State the target, retain source provenance and test each claimed behavior separately. |
| Adding named modules before locating a failure | More mechanisms can hide an encoding, objective or optimization problem. | First compare oracle actions, each task separately, joint learning and simple controls. |
| Calling supplied routing learned specialization | Manually assigned coordinate groups and ports already impose roles. | Use [fixed connectivity](partitioned.md) as an explicit experimental control; compare dense and disconnected models and count changes to capacity and initialization. |
| Concatenating blocks of different scale into one record reading | The larger block owns the address: the code barely moves with the smaller block, a few hub cells take most writes, and the store holds only its last writers. | Give every unit of the reading unit variance; measure cells in use, the share of the most active cells, and the code overlap when only one block changes ([record patch](record-patch.md)). |
| Normalising a record reading to unit norm | The cells' fixed offsets are in drive units; a small reading lets them decide, and the same cells win for every reading. | Keep unit variance per unit so a drive has unit scale; recount the cells in use after any change to the reading. |
| Reading a high record write rate as a memory of the corpus | At a rate near one half a cell holds its last few writers. Frozen records then recall little of the training stream, and reads on a new stream look calibrated only because its first writes overwrote the cells. | Score held-out data with records frozen, zeroed, writing online and writing from empty; use count-based averaging where retention matters and accept slower adaptation. |
| Letting a homeostatic offset measure its own scale | If the drive scale that bounds the offsets includes the offsets, the bound grows with them and cell usage oscillates. | Measure the scale without the offsets and bound the error term. |
| Expecting a rare boundary reading to be learned | A reading met once per stream gets a handful of updates per epoch and shares its record cells with common readings. | Make the boundary a common reading by convention, or present it as often as it matters; inspect the first event of a rollout from silence. |
| Sampling content from scores at positions the model cannot predict | Where held-out accuracy is at chance the scores carry no information: a draw is noise, and consecutive draws are chaos. | Measure accuracy at those positions separately; let the model decide when something happens there and declare what may happen. |
| Tuning a measurement's prior until the data show the expected structure | The model's score rises because the prior wrote the labels, while the labelled events lose their relation to the source. | Audit the fixture before training on it: faithfulness to the source against plain labelling, and whether the events of interest sit above chance where they should. |
| Launching a longer run because a short run converges | Settling may optimize an inadequate objective, and more steps can worsen interference. | Require acquisition, retention and actual task quality before scaling. |
| Expecting records to recall by position | A record is keyed by the event heard and the context. After a phase shift the same place in a bar hears another event, and the code overlap there falls from 0.91 to 0.16. | Put position in the reading as an input of its own, with enough ports to move the code, or measure the code overlap at the moments that should recall each other. |

For music, preserve stable clip IDs, the model and instrument versions, and
which observations or target records were available during generation. An
oracle render intentionally receives the target and must be labeled as a
control. A demo that improves only the instrument can still be useful, but
does not show better learning. Start with a bar that the instrument can
faithfully execute, then test coherent variation and longer intentions.

The [task-design guide](task-design.md) turns these checks into a repeatable
workflow. The [architecture guide](architecture.md) distinguishes implemented
capabilities from the continuing research target.
