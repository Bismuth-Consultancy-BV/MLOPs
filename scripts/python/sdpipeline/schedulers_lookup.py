from diffusers import DDIMScheduler,DDPMScheduler,DEISMultistepScheduler,DPMSolverMultistepScheduler,DPMSolverSinglestepScheduler,EulerAncestralDiscreteScheduler,EulerDiscreteScheduler,PNDMScheduler,UniPCMultistepScheduler,LMSDiscreteScheduler

schedulers = {
	"unipc" : UniPCMultistepScheduler,
	"euler_discrete" : EulerDiscreteScheduler,
	"euler_ancestral" : EulerAncestralDiscreteScheduler,
	"lms_discrete" : LMSDiscreteScheduler,
	"ddim" : DDIMScheduler,
	#"DDIMInverse" : DDIMInverseScheduler,
	"DDPM" : DDPMScheduler,
	"DEIS_Multistep" : DEISMultistepScheduler,
	"DPM_Multistep" : DPMSolverMultistepScheduler,
	"DPM_Singlestep" : DPMSolverSinglestepScheduler,
	#"HeunDiscrete" : HeunDiscreteScheduler,
	#"IPNDM" : IPNDMScheduler,
	#"KDPM2Ancestral" : KDPM2AncestralDiscreteScheduler,
	#"KDPM2" : KDPM2DiscreteScheduler,
	#"KarrasVe" : KarrasVeScheduler,
	"PNDM" : PNDMScheduler,
	#"RePaint" : RePaintScheduler,
	#"UnCLIP" : UnCLIPScheduler,
	#"VQDiffusion" : VQDiffusionScheduler
}