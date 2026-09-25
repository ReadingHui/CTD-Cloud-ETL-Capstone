# Reflection

## Prefect flow
The pipeline ran cleanly on the first try, everything loaded successfully, records enriched and upserted to the Supabase. The prefect UI showed `Completed` status on all the tasks and the flow itself, with all the printed messages as log in the flow. None of the tasks retried.

## LLM summaries
Most of the summaries reflects the weather features and the model's prediction accurately, but it struggle more when the cofidence is low. 

A particularly good one would be "With a high of 25.3°C, low of 18.7°C, minimal precipitation, and light winds, today is an excellent day for running." on 2023-09-17, which gives concrete values and correct analysis. 

A particularly bad one is "Despite favorable temperatures and no precipitation, the moderate wind speed suggests today may not be ideal for running." on 2023-04-27, as the summary and the choice appears to be contradicting. All the descriptions are saying it should be a good running day, but the result is not.

The reason could be this is one of those misclassified case, as the prediction model does not generate 100% accuracy, there could be times where all the weather data seems to be good for running, but the classification says otherwise. In that case, the LLM cannot make sense of the result no matter how hard it tries. Also, this particular example also only has a confidence of 0.4987, which is highly likely that it got misclassified.

## Daily scheduled deployment
I would need to change the `OPEN_METEO_CONFIG` to pull data from previous day's forecast instead of the preset range in 2023. That should be enough to keep the pipeline running smoothly on previous day's weather. However, it is also a good idea to update the prediction model once in a while, say every quarter or so, with the new and updated weather data.