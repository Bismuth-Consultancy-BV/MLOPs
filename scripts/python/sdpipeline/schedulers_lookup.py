from diffusers import HeunDiscreteScheduler,KDPM2DiscreteScheduler,KDPM2AncestralDiscreteScheduler,DDIMScheduler,DDPMScheduler,DEISMultistepScheduler,DPMSolverMultistepScheduler,DPMSolverSinglestepScheduler,EulerAncestralDiscreteScheduler,EulerDiscreteScheduler,UniPCMultistepScheduler,LMSDiscreteScheduler

schedulers = {
	"unipc" : UniPCMultistepScheduler,
	"euler" : EulerDiscreteScheduler,
	"euler a" : EulerAncestralDiscreteScheduler,
	"LMS" : LMSDiscreteScheduler,
	"Heun" : HeunDiscreteScheduler,
	"Heun Karras" : HeunDiscreteScheduler,
	"ddim" : DDIMScheduler,
	#"DDIMInverse" : DDIMInverseScheduler,
	"DDPM" : DDPMScheduler,
	"DEIS_Multistep" : DEISMultistepScheduler,
	"DPM2 Karras" : KDPM2DiscreteScheduler,
	"DPM2 a Karras" : KDPM2AncestralDiscreteScheduler,
	"DPM++ 2S" : DPMSolverSinglestepScheduler,
	# No argument to "turn" the following into Karras version yet
	# "DPM++ 2S Karras" : DPMSolverSinglestepScheduler(use_karras_sigmas=True),
	"DPM++ 2M" : DPMSolverMultistepScheduler,
	"DPM++ 2M Karras" : DPMSolverMultistepScheduler,
	#"IPNDM" : IPNDMScheduler,
	#"KarrasVe" : KarrasVeScheduler,
	#"PNDM" : PNDMScheduler,
	#"RePaint" : RePaintScheduler,
	#"UnCLIP" : UnCLIPScheduler,
	#"VQDiffusion" : VQDiffusionScheduler
}