[21st feb]
	discussion
		Step 1:	Clean DSTC
			for template

		Step 2:
			Multiple tasks

		Baselines:
			1. 2017 paper - upper limit.
			2. Lower limit - simple memnn using cleaned up data.
			3. Seq2Seq models are not as good as memnn, so we don't implement them.

		-- Interesting question: LSTMs vs MemNN.

		Template induction:
			1. understand code.
			2. email dipanjan das for code. 

			Data for templates:
				try subreddit for getting moar templates.

				Multilabel NER - template induction would work and provide a single field.
				Even if we want to get templates where each slot can have only one field, 
					where do we get such data???

		Transactional conversations:
			We dont have any.

		Once we have templates, we have to do 2 things:
			JUGAAD:
				- create a model that picks a template given {u_i}
					= can use lower
				- slot filling code.

			- IDEAL: joint inference? slot filling + template picking at the same time.
				= entity extraction
				= intent extraction
				C&C paradigm...
					= microsoft at interspeech conf. ... entity intent joint modeling. (we're doing the exact opposite.)
					= still read about it. get some ideas.

	todo:
		HH:
			0. read template induction... get code.
			1. hpc set up. (tomorrow) - TF + tutorials...
			2. cleanup code onto repo...
			3. look at dstc data.
		DR:
			1. start using the repo.
