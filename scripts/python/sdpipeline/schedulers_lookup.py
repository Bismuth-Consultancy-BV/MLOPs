from diffusers import EulerDiscreteScheduler, EulerAncestralDiscreteScheduler, LMSDiscreteScheduler, DDIMScheduler, UniPCMultistepScheduler

schedulers = {
	"unipc" : UniPCMultistepScheduler,
	"euler_discrete" : EulerDiscreteScheduler,
	"euler_ancestral" : EulerAncestralDiscreteScheduler,
	"lms_discrete" : LMSDiscreteScheduler,
	"ddim" : DDIMScheduler
}