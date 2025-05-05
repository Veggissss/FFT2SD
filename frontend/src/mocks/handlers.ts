import { http, HttpResponse } from "msw";

const KLINISK_MOCK = {
    "id": "16773557482994809422",
    "is_diagnose": false,
    "report_type": "klinisk",
    "text": "3 mm polypp, oppfattet som hyperplastisk.\nCa 15 cm fra anus åpning (rektum/sigmo overgang).\nCRC i familien.\n\nAnatomisk lokalisasjon: \n1/1: sigmoideum\nRekvirent bekrefter at korrekt hastegrad er valgt\nEr prøven tatt i forbindels med tarmscreeningsprogrammet?: Ja"
}

const MAKROSKOPISK_MOCK = {
    "id": "16773557482994809422",
    "is_diagnose": false,
    "report_type": "makroskopisk",
    "text": "Mulig bredbaset polypp fra sigmoideum, 3x3x2 mm, hel i #1"
}

const MIKROSKOPISK_MOCK = {
    "id": "6413961462004858133",
    "is_diagnose": false,
    "report_type": "mikroskopisk",
    "text": "Tubulovilløst adenom med lavgradig dysplasi, colonbiopsi."
}

const DIAGNOSE_MOCK = {
    "id": "4084497930299564480",
    "is_diagnose": true,
    "report_type": "mikroskopisk",
    "text": "1: Ileumslimhinne med lange slanke totter og regelmessig kryptarkitektur. Upåfallende sylinderepitel. Vanlig mengde og fordeling av betennelsesceller i lamina propria, inkludert noen prominente lymfoide aggregat, slik man normalt kan finne i denne lokasjon. Ikke sett inflammasjon, granulomer eller dysplasi.\n\n2- 4 (prøve 2- 3): Mange biter colonslimhinne med polypoid fasong. Her ses tettliggende tubul\u00e6re og villøse strukturer, kledd av sylinderepitel med lett til moderat begercelletap og lett forstørrede, lett hyperkromatiske, basalstilte kjerner. Utseende overveiende som tubulovilløst adenom med lavgradig dysplasi. I snitt 4, fra prøve 3, ses et lite område med kribriform vekst, uttalt begercelletap, og betydelig cytologisk atypi med avrundede kjerner som viser tap av polaritet. Funnene passer med et lite foci med høygradig dysplasi."
}

export const handlers = [
    http.post(`*/load_model`, () => {
        return HttpResponse.json(
            { "success": true, "message": "MOCKED model loaded successfully! (Its not really loaded...)" }
        )
    }),

    http.get(`*/unlabeled/auto`, () => {
        return HttpResponse.json(
            KLINISK_MOCK
        )
    }),

    http.get(`*/unlabeled/klinisk`, () => {
        return HttpResponse.json(
            KLINISK_MOCK
        )
    }),

    http.get(`*/unlabeled/makroskopisk`, () => {
        return HttpResponse.json(
            MAKROSKOPISK_MOCK
        )
    }),

    http.get(`*/unlabeled/mikroskopisk`, () => {
        return HttpResponse.json(
            MIKROSKOPISK_MOCK
        )
    }),

    http.get(`*/unlabeled/diagnose`, () => {
        return HttpResponse.json(
            DIAGNOSE_MOCK
        )
    }),


    http.post(`*/generate`, () => {
        return HttpResponse.json([
            {
                "input_text": "Mulig bredbaset polypp fra sigmoideum, 3x3x2 mm, hel i #1",
                "metadata_json": [
                    {
                        "enum": [
                            {
                                "value": "klinisk"
                            },
                            {
                                "value": "makroskopisk"
                            },
                            {
                                "value": "mikroskopisk"
                            }
                        ],
                        "field": "Rapport type",
                        "id": 0,
                        "type": "enum",
                        "value": "makroskopisk"
                    },
                    {
                        "field": "Antall glass",
                        "id": 1,
                        "type": "int",
                        "unit": "stk",
                        "value": 1
                    },
                    {
                        "field": "Beholder-ID",
                        "id": 128,
                        "type": "int",
                        "value": 1
                    }
                ],
                "target_json": [
                    {
                        "enum": [
                            {
                                "name": "terminale ileum",
                                "value": "T65520"
                            },
                            {
                                "name": "coecum",
                                "value": "T67100"
                            },
                            {
                                "name": "colon ascendens",
                                "value": "T67200"
                            },
                            {
                                "name": "høyre colonfleksur",
                                "value": "T67300"
                            },
                            {
                                "name": "colon transversum",
                                "value": "T67400"
                            },
                            {
                                "name": "venstre colonfleksur",
                                "value": "T67500"
                            },
                            {
                                "name": "colon descendens",
                                "value": "T67600"
                            },
                            {
                                "name": "colon sigmoideum",
                                "value": "T67700"
                            },
                            {
                                "name": "rectosigmoid",
                                "value": "T68200"
                            },
                            {
                                "name": "rectum",
                                "value": "T68000"
                            },
                            {
                                "name": "anus",
                                "value": "T69000"
                            },
                            {
                                "name": "colon",
                                "value": "T67000"
                            },
                            {
                                "name": "colon og rectum",
                                "value": "T67920"
                            }
                        ],
                        "field": "Lokasjon patologi",
                        "id": 107,
                        "type": "enum",
                        "value": "T67700"
                    },
                    {
                        "enum": [
                            {
                                "name": "endoskopisk biopsi",
                                "value": "P13400"
                            },
                            {
                                "name": "endoskopisk slyngereseksjon",
                                "value": "P13402"
                            },
                            {
                                "name": "endoskopisk slyngereseksjon piecemeal",
                                "value": "P13403"
                            },
                            {
                                "name": "endoskopisk submucosal disseksjon",
                                "value": "P13405"
                            },
                            {
                                "name": "endoskopisk materiale, prosedyre ukjent",
                                "value": "P13409"
                            },
                            {
                                "name": "konsultasjonspreparat",
                                "value": "P2070A"
                            }
                        ],
                        "field": "Prøvemateriale",
                        "id": 108,
                        "type": "enum",
                        "value": "P13409"
                    },
                    {
                        "field": "Makroskopisk beskrivelse",
                        "id": 109,
                        "type": "string",
                        "value": "3x3x2 mm, hel i #1"
                    }
                ]
            }
        ])
    }),

    http.post(`*/correct/:reportId`, () => {
        return HttpResponse.json(
            { "message": "Correctly MOCK labeled JSON!" }
        )
    }),

    http.get(`*/models`, () => {
        return HttpResponse.json(
            {
                "decoder": [
                    "norallm/normistral-7b-warm_4bit_quant",
                    "norallm/normistral-7b-warm",
                    "google/gemma-3-27b-it",
                    "Qwen/Qwen3-32B"
                ],
                "encoder": [
                    "ltg/norbert3-small_mask_values",
                    "ltg/norbert3-small",
                    "ltg/norbert3-base",
                    "ltg/norbert3-large"
                ],
                "encoder_decoder": [
                    "ltg/nort5-small",
                    "ltg/nort5-base",
                    "ltg/nort5-large"
                ]
            }
        )
    })
];