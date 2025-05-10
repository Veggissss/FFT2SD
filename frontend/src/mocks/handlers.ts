import { http, HttpResponse } from "msw";
import unlabeled_klinisk_mock from "./data/unlabeled_endpoint/klinisk_mock.json";
import unlabeled_makroskopisk_mock from "./data/unlabeled_endpoint/makroskopisk_mock.json";
import unlabeled_mikroskopisk_mock from "./data/unlabeled_endpoint/mikroskopisk_mock.json";
import unlabeled_diagnose_mock from "./data/unlabeled_endpoint/diagnose_mock.json";

import generated_klinisk_mock from "./data/generate_endpoint/klinisk_mock.json";
import generated_makroskopisk_mock from "./data/generate_endpoint/makroskopisk_mock.json";
import generated_mikroskopisk_mock from "./data/generate_endpoint/mikroskopisk_mock.json";
import generated_diagnose_mock from "./data/generate_endpoint/diagnose_mock.json";

export const handlers = [
    http.post(`*/load_model`, () => {
        return HttpResponse.json({
            success: true,
            message: "MOCKED model loaded successfully! (Its not really loaded...)"
        });
    }),

    http.get(`*/unlabeled/:type`, ({ params }) => {
        const type = params.type as string;
        switch (type) {
            case 'auto':
            case 'klinisk':
                return HttpResponse.json(unlabeled_klinisk_mock);
            case 'makroskopisk':
                return HttpResponse.json(unlabeled_makroskopisk_mock);
            case 'mikroskopisk':
                return HttpResponse.json(unlabeled_mikroskopisk_mock);
            case 'diagnose':
                return HttpResponse.json(unlabeled_diagnose_mock);
            default:
                return new HttpResponse(null, { status: 404 });
        }
    }),


    // Generation endpoint
    http.post(`*/generate`, async ({ request }) => {
        const body = await request.json() as Record<string, any>;
        if (!body || !body.report_type || !body.input_text) {
            return HttpResponse.json(
                { "error": "Invalid request" },
                { status: 400 }
            );
        }
        const reportType: string = body.report_type;
        const inputText: string = body.input_text;

        // Mock data to return based on report type
        let responseData;
        switch (reportType.toLowerCase()) {
            case 'klinisk':
                responseData = generated_klinisk_mock;
                break;
            case 'makroskopisk':
                responseData = generated_makroskopisk_mock;
                break;
            case 'mikroskopisk':
                responseData = inputText === unlabeled_diagnose_mock.text ? generated_diagnose_mock : generated_mikroskopisk_mock;
                break;
            default:
                responseData = { "error": `Unknown report type: ${reportType}` };
        }
        return HttpResponse.json(responseData);
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