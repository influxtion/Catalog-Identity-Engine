"""One separately registered conventional matcher, sharing the fixed evaluation protocol."""

from catalog_identity.evaluation import cross_encoder as runner
from catalog_identity.matching.full_encoder import ARCHITECTURE, TRAINING, FullMatcher

FULL_PROTOCOL = runner.JointProtocol(
    architecture=ARCHITECTURE,
    training=TRAINING,
    matcher=FullMatcher,
    screening_limit=1800,
    full_fit_limit=1200,
    allowed_microbatches=(8,),
    source_files=runner.JOINT_SOURCE_FILES
    + (
        "matching/full_encoder.py",
        "evaluation/full_encoder.py",
    ),
    report_prefix="full_encoder",
    measure_full_preparation=True,
    freeze_validation_policy=True,
)


def pilot(config, output, validation_report, device="cpu", matcher_factory=None):
    return runner.pilot(
        config,
        output,
        validation_report,
        matcher_factory=matcher_factory,
        device=device,
        protocol=FULL_PROTOCOL,
    )


def screen(config, output, timing, control, device="cpu"):
    return runner.screen_joint(config, output, timing, control, device, protocol=FULL_PROTOCOL)


def develop(config, output, screening, device="cpu"):
    return runner.develop_joint(config, output, screening, device, protocol=FULL_PROTOCOL)
