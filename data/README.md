# TED Multilingual Corpora


## Content 
- `utils`: Scripts that were used to crawl ted talk transcripts in data/tedtalks

- `tedtalks`: Transcripts of 3677 TED talks obtained by running the code under `utils`. 

Note that in our experiments we only use 1518 of them for which we have transcripts in all four languages. (DE, ES, FR, TR)

- `TED_annotations.csv`: Meta information for each TED Talk in the `tedtalks` directory

- `files.tsv`: File names of 1518 TED talks we use in our experiments.

- `prons.csv`: List of gendered pronouns we used during gender annotation.

**Important** : If you would like to repeat experiments described in the paper, you can leave this directory as it is. The code
under `adv/src` will only use the 1518 TED talks listed in `files.tsv`

## Licensing

TED licenses its materials for non-profit use under the Creative Commons license, Attribution–Non Commercial–No Derivatives (or the CC BY – NC – ND 4.0 International) which means it may be shared by following a few requirements specified by the TED organization:

* Attribute TED as the owner of the TED Talk and include a link to the talk, but do not include any text that shows TED endorses your website or platform.
* Do not use the TED site content for any commercial purposes, for sale, sublicense or in an app of any kind for any advertising, or in exchange for payment of any kind.
* You cannot remix, create derivative work or modify the TED site content in any way.
* You may not add any further more restrictions that we have provided to the TED site content, such as additional legal restrictions, or charge any fees or technological restrictions to accessing the content.


To view a copy of the BY-NC-ND license, visit http://creativecommons.org/licenses/by-nc-nd/4.0/ or send a letter to Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.

<a rel="license" href="http://creativecommons.org/licenses/by-nc-nd/4.0/"><img alt="Creative Commons License" style="border-width:0" src="https://i.creativecommons.org/l/by-nc-nd/4.0/88x31.png" /></a><br />This work is licensed under a <a rel="license" href="http://creativecommons.org/licenses/by-nc-nd/4.0/">Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International License</a>.

For more information on TED's licensing: https://www.ted.com/about/our-organization/our-policies-terms/ted-talks-usage-policy.
