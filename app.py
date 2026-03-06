import gradio as gr
import hydra
from omegaconf import DictConfig

from chess_classifier.data import fen_to_features
from chess_classifier.models import Model


@hydra.main(version_base=None, config_path="../configs", config_name="predict")
def main(cfg: DictConfig):
    model = Model.load(cfg.predict.model_path)

    def predict(fen, white_elo, black_elo):
        inp = fen_to_features(fen, white_rating=white_elo, black_rating=black_elo)

        ((white_win_prob, black_win_prob, draw_prob),) = model.predict_proba(inp)

        return dict(White=white_win_prob, Black=black_win_prob, Draw=draw_prob)

    # demo = gr.Interface(
    #     fn=predict,
    #     inputs=["text", "number", "number"],
    #     outputs=["label"],
    #     api_name="predict"
    # )

    with gr.Blocks() as demo:
        with gr.Row():
            with gr.Column(scale=1, min_width=550):  # fixed width for iframe
                gr.HTML(
                    """
                    <iframe src="https://zoravur.com/chessboard-editor/"
                            width="525" height="765"
                            style="border:none;">
                    </iframe>
                    <p>Copy the FEN from the editor above and paste to the right:</p>
                """
                )
            with gr.Column(scale=1, min_width=300):  # inputs column
                inp = [
                    gr.Text(label="FEN", placeholder="rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"),
                    gr.Number(label="White Elo", value=1500),
                    gr.Number(label="Black Elo", value=1500),
                ]
                out = gr.Label(label="Predicted Outcome")
                btn = gr.Button("Run")
                btn.click(fn=predict, inputs=inp, outputs=out)

    demo.launch(share=True)
    # with gr.Blocks() as demo:
    #     with gr.Row():
    #         gr.HTML("""
    #             <iframe src="https://zoravur.com/chessboard-editor/"
    #                     width="525" height="765"
    #                     style="border:none;">
    #             </iframe>
    #             <p>Copy the FEN from the editor above and paste to the right:</p>
    #         """)
    #         with gr.Column():
    #             inp = [
    #                 gr.Text(label="FEN", placeholder="rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"),
    #                 gr.Number(label="White Elo", value=1500),
    #                 gr.Number(label="Black Elo", value=1500)
    #             ]
    #     out = gr.Label(label="Predicted Outcome")
    #     btn = gr.Button("Run")
    #     btn.click(fn=predict, inputs=inp, outputs=out)

    # demo.launch(share=True)


if __name__ == "__main__":
    main()
