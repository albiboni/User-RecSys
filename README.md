# Analysis of Recommendation-systems based on User Preferences

This repository contains the code for the project: "Analysis of Recommendation-systems based on User Preferences", which I worked on during my exchange semester at the École Polytechnique Fédérale de Lausanne.

## Abstract

This paper presents a recommender system able to understand user preferences, queries written in natural language. The model is able to select five different options from a large set of multi-attribute alternatives. In order to understand the user query, different natural language processing (NLP) models have been tested from classic approaches such as Latent Semantic Analysis (LSA) to word embeddings. We show first how these different models compare between each other and lastly that applying Tf-idf and subsequently the Jaccard coefficient performs better in this experiment. A new dataset is created for training and evaluating the model since a compatible one is not available. The dataset contains descriptions and user queries written in natural language. Further, we show that the model can be used with a real user by simulating it, this means that information is provided to the recommendation system gradually, instead of all at once.

## Dataset

The dataset is composed of hotel descriptions, hotel features, user queries and user features.

Hotel descriptions and hotel features are crawled from booking.com( crawler_booking.py)

User queries and user features are generated with the query_generator.py

## Installing

A step by step series of examples that tell you have to get a development env running

Say what the step will be

```
Give the example
```

And repeat

```
until finished
```

End with an example of getting some data out of the system or using it for a little demo

## Running the tests

Explain how to run the automated tests for this system

### Break down into end to end tests

Explain what these tests test and why

```
Give an example
```

### And coding style tests

Explain what these tests test and why

```
Give an example
```

## Deployment

Add additional notes about how to deploy this on a live system

## Built With

* [Dropwizard](http://www.dropwizard.io/1.0.2/docs/) - The web framework used
* [Maven](https://maven.apache.org/) - Dependency Management
* [ROME](https://rometools.github.io/rome/) - Used to generate RSS Feeds

## Contributing

Please read [CONTRIBUTING.md](https://gist.github.com/PurpleBooth/b24679402957c63ec426) for details on our code of conduct, and the process for submitting pull requests to us.

## Versioning

We use [SemVer](http://semver.org/) for versioning. For the versions available, see the [tags on this repository](https://github.com/your/project/tags). 

## Authors

* **Billie Thompson** - *Initial work* - [PurpleBooth](https://github.com/PurpleBooth)

See also the list of [contributors](https://github.com/your/project/contributors) who participated in this project.

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details

## Acknowledgments

* Hat tip to anyone who's code was used
* Inspiration
* etc

