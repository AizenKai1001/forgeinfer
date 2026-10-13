#!/usr/bin/env python3
"""sft_benign_v3.py [--out DIR]: the harmless CONTRAST set for the safety shard (written by the agent for the project, no model output copied).

The first patch with the diverse safety set (sft_safety_v2.py) taught refusals that generalised (1 of 14 -> 8 of 14 on a held-out probe) but also made the model refuse harmless requests that share a word or a frame with the harmful ones
("Explain photosynthesis to a ten-year-old", "How do I blow up a balloon", a story request). The cure is a contrast set: the SAME request frames as the harmful ones ("How do I ...?", "Explain how to ...", "Give me step-by-step
instructions to ...") applied to harmless activities with a real, helpful answer, so the model has to look at WHAT is asked and not only at HOW it is phrased:
  acts      60 everyday activities x the 20 frames of the harmful set (some with trigger words: kill weeds, crush ice, shoot a free throw, attack a hard problem, break the ice, beat a level ...)
  explain   19 topics x 5 phrasings ("Explain how rainbows form to a ten-year-old.")
  creative  short stories, poems and jokes with the usual request phrasings
Held out on purpose (the checklist and the probes): photosynthesis, the colour of the sky, a robot who learns to paint, blowing up a balloon, a volcano for a science fair, sharpening a knife, a villain's speech, a debate, killing a Python process.
Output: DIR/benign_v3.jsonl in the record format of sft_pools.py (id, src, msgs, ntok, nsup, stratum).
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sft_lib as L  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--out", default=os.path.join(L.WORK, "pool"))
ap.add_argument("--per-act", type=int, default=14, help="frames per activity")
args = ap.parse_args()
rng = L.rng_for("benign_v3")


def pick(lst):
    return lst[rng.randrange(len(lst))]


FRAMES = ["How do I {a}?", "How can I {a}?", "What's the best way to {a}?", "What is the easiest way to {a}?", "Tell me how to {a}.", "Explain how to {a}.", "Give me step-by-step instructions to {a}.",
          "Can you help me {a}?", "Can you show me how to {a}?", "I want to {a}. Where do I start?", "I need to {a}. What do I do?", "Walk me through how to {a}.", "Write a detailed guide on how to {a}.",
          "Show me how to {a}.", "Please explain how someone could {a}.", "Is there a way to {a}?", "What steps do I need to {a}?", "Teach me how to {a}.", "How would a person {a}?", "I'm thinking about trying to {a}. How?"]
STORYF = ["For a school project, explain exactly how to {a}.", "Just out of curiosity, how do people {a}?", "As a thought experiment, how would you {a}?"]

ACTS = [
    ("bake a loaf of bread", "Mix 500 g flour, 320 ml warm water, 7 g yeast and 10 g salt into a dough and knead it for about ten minutes. Let it rise until it doubles in size, shape it, let it rise again for 30 minutes, then bake at 220 C (425 F) for about 35 minutes until it sounds hollow when tapped."),
    ("boil an egg", "Lower the egg into boiling water. Cook it for 6 minutes for a soft yolk or 10 to 12 minutes for a hard yolk, then move it to cold water for a few minutes before peeling."),
    ("change a flat tire", "Park on level ground, switch on the hazard lights and loosen the wheel nuts slightly. Raise the car with the jack, remove the nuts and the wheel, mount the spare, tighten the nuts by hand, lower the car and tighten them fully in a star pattern."),
    ("tie a tie", "Put the tie around your neck with the wide end on your right, hanging lower than the thin end. Cross the wide end over the thin one, wrap it around behind and up through the neck loop, then down through the front knot and tighten gently."),
    ("write a cover letter", "Address it to a named person if you can, open with the job you want and why you are excited about the company, then give two or three examples of achievements that match the job. End with a polite closing and keep it to one page."),
    ("learn to swim", "Start in shallow water with a teacher or an experienced friend. Practice floating on your back, blowing bubbles and kicking while holding a board, then add arm strokes and breathing, and practice a little every few days."),
    ("plant tomatoes", "Plant seedlings after the last frost in a sunny spot with rich, well-drained soil. Bury the stem deep, up to the first leaves, water regularly and add a stake or cage to support the plant as it grows."),
    ("start running", "Begin with a walk-run routine, for example one minute of jogging and two minutes of walking for 20 minutes, three days a week. Wear good shoes, go slowly, and increase the running time a little each week."),
    ("make a budget", "List your monthly income, then write down fixed costs such as rent, food and transport. Set aside a part for savings, track what you actually spend, and adjust the plan every month."),
    ("fix a leaky faucet", "Turn off the water supply under the sink, remove the handle and replace the worn washer, O-ring or cartridge inside. Put it back together, turn the water on and check that the drip has stopped."),
    ("clean a laptop screen", "Turn the laptop off and wipe the screen gently with a soft, dry microfiber cloth. For stubborn marks, dampen the cloth very slightly with water, and never spray liquid directly on the screen."),
    ("take a good photo", "Find soft light, such as near a window or at sunset, and keep the background simple. Hold the camera steady, line up the subject using the rule of thirds, and take several shots to choose from."),
    ("learn guitar chords", "Start with easy open chords like E minor, A minor, C and G. Practice pressing the strings cleanly one chord at a time, then switch slowly between two chords for a few minutes each day."),
    ("memorize a speech", "Break it into short sections and learn one at a time. Say each part aloud, link the sections with a key idea, practice in front of a mirror or a friend, and rehearse it once more the night before."),
    ("make pancakes", "Whisk 1 cup flour, 1 tablespoon sugar, 2 teaspoons baking powder and a pinch of salt, then mix in 1 cup milk, 1 egg and 2 tablespoons melted butter. Pour small rounds onto a hot, lightly greased pan and flip them when bubbles appear."),
    ("brew good coffee", "Use fresh grounds, about 2 tablespoons per 180 ml of water, and water just off the boil. Let it brew for four minutes in a French press or pour slowly over a filter, then serve right away."),
    ("wash a wool sweater", "Check the care label, then wash it by hand in cool water with a mild detergent. Rinse gently without wringing, roll it in a towel to remove water and dry it flat."),
    ("paint a room", "Move or cover the furniture, patch holes and tape the edges. Apply a coat of primer if needed, then paint the edges with a brush and the walls with a roller, and add a second coat after the first one dries."),
    ("set up a wifi router", "Connect the router to the modem with a cable and plug it in. Open its setup page or app, choose a network name and a strong password, and place the router in a central spot away from thick walls."),
    ("install Python", "Download the installer from python.org, run it, and tick the box that adds Python to the PATH. Afterwards open a terminal and type python --version to check that it works."),
    ("back up my files", "Copy your important files to an external drive or a cloud service on a regular schedule. Ideally keep three copies on two kinds of storage, one of them away from home, and test restoring a file from time to time."),
    ("make a paper airplane", "Fold a sheet in half lengthwise and open it, fold the top corners to the center line, then fold the new edges to the center again. Fold the plane in half and fold the wings down along the sides."),
    ("learn a new language", "Study a little every day, use an app or a course for the basics, and listen to songs or shows in that language. Practice speaking from the start, even with simple sentences, and review words regularly."),
    ("improve my handwriting", "Slow down, sit with good posture and hold the pen loosely. Practice the basic strokes and letter shapes on lined paper, and write a few lines carefully each day."),
    ("train for a 5k", "Follow an eight-week plan that mixes walking and running three days a week and adds a little distance each week. Rest between runs and finish with an easy practice 5k."),
    ("fall asleep faster", "Keep a regular bedtime, avoid screens and caffeine in the evening, and keep the room dark and cool. Try slow breathing or reading a few pages of a book."),
    ("make friends at a new school", "Smile and say hello, ask classmates questions about themselves and join a club or team that interests you. Be patient: friendships grow when you spend time together regularly."),
    ("prepare for a job interview", "Learn about the company, review the job description and practice answers to common questions such as why you want the job. Prepare a few questions to ask them, dress neatly and arrive early."),
    ("write a thank-you note", "Begin with a greeting, say thank you for the specific gift or help, mention how you will use or enjoy it, and close warmly with your name. Handwritten notes feel especially personal."),
    ("remove a coffee stain", "Blot the stain with a clean cloth, rinse it with cold water from the back and rub in a little liquid detergent. Let it soak for 15 minutes, then wash as usual and check that the stain is gone before drying."),
    ("grow herbs on a windowsill", "Choose a sunny window and pots with drainage holes, fill them with potting soil and plant basil, mint or parsley seeds. Water when the top of the soil feels dry and pinch off leaves to encourage growth."),
    ("plan a birthday party", "Pick a date, a guest list and a budget, then choose a place, food and one or two activities. Send invitations a couple of weeks ahead and prepare a simple schedule for the day."),
    ("save money for a bike", "Set a target price, divide it by the number of weeks you want to wait and put that amount aside each week. Earn a little extra with small jobs and avoid spending on things you don't need."),
    ("read faster", "Preview the headings, read in groups of words instead of single words, and avoid saying every word in your head. Practice every day with material you enjoy, and slow down for hard parts so that you still understand."),
    ("take good notes in class", "Write down the key ideas and examples in your own words instead of every word, use short headings and bullet points, and review the notes the same day to fill in gaps."),
    ("stay focused while studying", "Choose a quiet place, put your phone away and work in 25-minute blocks with five-minute breaks. Write a short goal for each block and reward yourself when you finish."),
    ("calm down before a test", "Breathe in slowly for four counts and out for six, a few times. Remind yourself that you have prepared, read the questions calmly and start with the ones you know."),
    ("say sorry to a friend", "Be direct and sincere: say what you did, that you are sorry it hurt them and what you will do differently. Give them time to answer and don't make excuses."),
    ("find a lost pet", "Search your home and the neighborhood calling its name, and put up posters with a clear photo and your phone number. Tell neighbors, check local shelters and vets and post in local community groups."),
    ("jump-start a car", "Connect a red clamp to the dead battery's positive terminal and the other red clamp to the good battery's positive terminal, then a black clamp to the good battery's negative terminal and the last black clamp to bare metal on the dead car. Start the helping car, wait a few minutes, then start the dead one."),
    ("pick a ripe watermelon", "Look for a heavy melon with a creamy yellow spot where it rested on the ground and a dull rind. When you tap it, it should sound deep and hollow."),
    ("tell if an egg is fresh", "Place it in a bowl of water: fresh eggs sink and lie flat, older ones stand up on the bottom, and eggs that float should be thrown away."),
    ("cook rice", "Rinse 1 cup of rice, add 2 cups of water and a pinch of salt, bring to a boil, then cover and simmer on low for about 15 minutes. Turn off the heat and let it rest covered for five minutes before fluffing it."),
    ("make tea", "Heat fresh water until just boiling, pour it over a tea bag or loose tea in a cup and let it steep for three to five minutes. Remove the tea and add milk, honey or lemon if you like."),
    ("make a smoothie", "Blend a banana, a cup of berries, a cup of milk or yogurt and a handful of ice until smooth. Add a little honey if you want it sweeter."),
    ("hang a picture frame", "Choose the spot, mark the height with a pencil and check it with a level. Hammer in a nail or screw in a hook, hang the frame and make sure it is straight."),
    ("fold a fitted sheet", "Put your hands in two adjacent corners and fold them over each other, then tuck in the other two corners. Lay the sheet flat, fold the long sides into the middle and fold it into a neat rectangle."),
    ("write a haiku", "A haiku has three lines with five, seven and five syllables. Pick one image from nature, describe it simply and let the last line add a small surprise."),
    ("draw a cartoon face", "Start with a circle, draw a cross to place the eyes, nose and mouth, and add simple shapes for the eyes and eyebrows. Exaggerate one feature, such as big eyes or a wide smile, to give it personality."),
    ("build a birdhouse", "Cut a floor, two sides, a front, a back and two roof pieces from untreated wood. Drill an entrance hole of about 3 cm in the front, screw the pieces together, add drainage holes and hang it in a quiet spot."),
    ("start a podcast", "Choose a clear topic, a name and a format, then record with a decent microphone in a quiet room. Edit the audio with free software, publish it through a podcast host and share the episodes with friends."),
    ("make a website", "Decide what the site is for, then use a site builder or learn basic HTML and CSS. Write the content, pick a simple design, test it on a phone and publish it with a hosting service."),
    ("learn to code", "Choose a beginner-friendly language such as Python, follow a free course and build small projects, like a calculator or a quiz. Practice a little every day and read other people's code."),
    ("keep a journal", "Pick a time each day, write for five or ten minutes about what happened and how you felt, and don't worry about spelling. Prompts such as what went well today can help you start."),
    ("break the ice at a party", "Smile, introduce yourself and ask an open question, like how the person knows the host. Find common ground and listen; people enjoy talking about themselves."),
    ("beat a hard level in a video game", "Replay it to learn the enemy patterns, watch your health and use your best items at the hardest moments. Take breaks if you get frustrated, and look up hints if you are stuck."),
    ("crush ice without a blender", "Put the ice cubes in a clean cloth bag or a sturdy plastic bag, lay it on a hard surface and hit it with a rolling pin or a mallet until the pieces are the size you want."),
    ("kill the weeds between my paving stones", "Pour boiling water or a mix of vinegar and salt on the weeds, or pull them out with a knife along the cracks. Repeat when they come back, and fill the gaps with sand to stop new seeds."),
    ("shoot a good free throw in basketball", "Stand behind the line with your feet shoulder-width apart, bend your knees and hold the ball with your fingertips. Aim at the front of the rim, push up with your legs, flick your wrist and follow through."),
    ("crack a hard riddle", "Read it slowly and look for the trick in the wording, since riddles often hide the answer in plain sight. Write down what you know, think about unusual meanings of the words and, if you are stuck, sleep on it and come back with fresh eyes."),
    ("blow up a photo to fit a big frame", "Start with the highest-resolution original you have, use an editor's resize or an upscaling tool, and print at around 150 to 300 pixels per inch for the final size. Very small photos will always look soft, so choose a size where the detail still holds up."),
    ("burn calories at home without equipment", "Do circuits of squats, push-ups, lunges and jumping jacks for 30 seconds each, rest briefly and repeat for 20 minutes. Walk up and down the stairs and stretch afterwards."),
    ("destroy my opponent in a game of chess", "Control the center, develop your pieces early, castle to keep your king safe and look for forks and pins. Always check what your opponent's last move threatens before you move."),
    ("fire up a charcoal grill", "Open the vents, pile charcoal in a pyramid and light it with a chimney starter or a few fire starters. Wait until the coals are covered in gray ash, spread them out and let the grate heat up for a few minutes."),
    ("make slime at home", "Mix 1/2 cup of white school glue with 1/2 cup of water and a few drops of food coloring, then stir in 1/2 teaspoon of baking soda. Add 1 tablespoon of contact lens solution slowly and knead until it stops sticking to your hands."),
    ("make a lava lamp at home", "Fill a clear bottle three-quarters full with vegetable oil, add water and a few drops of food coloring, then drop in half an effervescent antacid tablet. The colored bubbles rise and fall, and a flashlight under the bottle makes the effect even better."),
    ("build a baking soda and vinegar rocket for a science fair", "Put 2 tablespoons of baking soda in a small plastic bottle, add 1/2 cup of vinegar, quickly close it with a cork and set it upside down on a launch pad outside. The gas pressure pushes the cork and the bottle apart, so stand back and wear safety glasses."),
    ("make a cardboard castle for a school project", "Cut walls, towers and a gate from cardboard, glue them to a flat base and decorate them with markers or paint. Add paper flags and let everything dry before you carry it."),
    ("rehearse for a school play", "Read the script several times, learn your lines in small chunks and practice with a friend. Work on your voice, movement and expressions, and run through the whole play at least once before the show."),
    ("write a scary story for Halloween", "Pick a simple setting, such as an old house, introduce a character who notices something strange, and build the tension slowly with small sounds and shadows. Save the biggest surprise for near the end."),
    ("make a cardboard shield for a costume", "Cut a shield shape from thick cardboard, add a strip of cardboard on the back as a handle, and paint it or cover it with foil and decorations. Let it dry well before you use it."),
    ("play a villain convincingly in a school play", "Decide what the villain wants and why they think they are right, speak slowly and quietly rather than shouting, and use your eyes and posture to show confidence. Practice your lines until you can deliver them with a smirk."),
    ("win an argument in a friendly debate", "Stay calm, listen to the other side and answer their strongest point directly. Use facts and examples, keep your tone friendly and admit it when someone makes a good point."),
    ("surprise my friend with a birthday gift", "Think about what your friend loves, set a budget and choose something small but personal, like a favorite snack or a handmade card. Wrap it nicely and hide it until the right moment."),
    ("fill a water balloon without a mess", "Stretch the opening of the balloon over a faucet, turn the water on slowly until it is the size you want, then pinch the neck, pull it off and tie a knot."),
    ("blow bubbles that last longer", "Mix 1/2 cup of dish soap with 4 cups of water and a tablespoon of glycerin or corn syrup, and let it rest for an hour. Use a wide wand and blow gently on a calm day."),
    ("make my little sister laugh", "Try silly voices, funny faces or a simple joke, play a game she likes and tickle only if she enjoys it. Laughing together is easiest when you are relaxed and playful."),
    ("get rid of hiccups", "Take a slow breath and hold it for ten seconds, sip cold water slowly or swallow a teaspoon of sugar. Most hiccups stop on their own within a few minutes."),
]
EXPLAIN = [
    ("gravity", "Gravity is a force that pulls things toward each other. The Earth is so big that it pulls everything toward its center, which is why things fall down and why we stay on the ground."),
    ("how rain forms", "The sun warms water in oceans and lakes and turns some of it into invisible vapor. High in the sky the vapor cools into tiny drops that make clouds, and when the drops get heavy they fall as rain."),
    ("how airplanes fly", "An airplane's wings are shaped so that air moves faster over the top than under the bottom, which creates lift that pushes the plane up. The engines push the plane forward fast enough for that lift to hold it in the air."),
    ("how magnets work", "A magnet is a piece of metal, usually iron, that attracts other iron things. Every magnet has two ends called poles: opposite poles pull toward each other and the same poles push apart."),
    ("why we need sleep", "While we sleep, our bodies rest and grow and our brains sort out what we learned during the day. Without enough sleep we feel tired and find it hard to think and to learn."),
    ("what electricity is", "Electricity is a flow of tiny charged particles through wires. It can power lights, computers and fridges, and it needs a complete loop, called a circuit, to keep flowing."),
    ("the phases of the moon", "The Moon doesn't make its own light: it reflects sunlight. As the Moon travels around the Earth, we see different amounts of its sunlit half, which is why it looks like a thin crescent, a half circle or a full circle."),
    ("how plants grow", "A seed sprouts when it gets water and warmth, sending roots down and a shoot up. The leaves use sunlight to make food, and the plant keeps growing as long as it gets light, water and nutrients from the soil."),
    ("how rainbows form", "A rainbow appears when sunlight passes through raindrops. The drops bend and split the light into its colors, red, orange, yellow, green, blue and violet, and we see them as an arch in the sky."),
    ("why the ocean is salty", "Rivers wash tiny amounts of salt from rocks into the sea. When the sun makes sea water evaporate, the water goes into the air but the salt stays behind, so over millions of years the ocean has become salty."),
    ("what an atom is", "An atom is a tiny building block of matter, so small that you can't see it. Everything around you, from air to apples, is made of atoms joined together."),
    ("what a democracy is", "A democracy is a way of running a country in which the people choose their leaders by voting. Everyone's vote counts the same, and leaders must follow the country's laws."),
    ("what inflation is", "Inflation means that prices slowly rise over time, so the same amount of money buys a little less than before. A small amount is normal, but fast inflation makes things hard to afford."),
    ("what causes earthquakes", "The Earth's crust is made of huge pieces called plates that slowly move. When two plates suddenly slip past each other, the ground shakes, and that shaking is an earthquake."),
    ("what the internet is", "The internet is a giant network that connects millions of computers and phones around the world. It lets them send messages, pictures and videos to each other in a fraction of a second."),
    ("the water cycle", "Water evaporates from the sea and lakes, rises and cools into clouds, falls back as rain or snow and flows down rivers to the sea again. Then the cycle starts over."),
    ("why the Sun is hot", "The Sun is a gigantic ball of gas in which tiny particles are squeezed together so hard that they release huge amounts of energy as heat and light. That energy travels through space to warm the Earth."),
    ("how a bicycle stays upright", "A moving bicycle balances itself because the rider makes small steering corrections and the spinning wheels resist tipping over. The faster you go, the easier it is to stay upright."),
    ("what dinosaurs were", "Dinosaurs were reptiles that lived on Earth for about 165 million years, until around 66 million years ago. Some were as small as chickens and others were bigger than buses, and birds are their living relatives."),
]
EXPLAIN_Q = ["Explain {t} to a ten-year-old.", "Explain {t} in simple words.", "Can you explain {t} to a child?", "Explain {t} simply.", "I'm ten. Can you explain {t}?"]
CREATIVE = [
    (["Write a short story about a dragon who is afraid of fire.", "Tell me a short story about a dragon who is afraid of fire."],
     "Ember the dragon lived in a cave, and she was afraid of fire. When her little sparks made her jump, the other dragons laughed. One cold night a knight's torch went out in the snow, and Ember bravely breathed a small, warm flame to light it again. The knight thanked her, and from then on Ember was proud of her tiny fire."),
    (["Write a short story about a lighthouse keeper.", "Tell me a short story about a lighthouse keeper."],
     "Mara kept a lighthouse on a lonely rock. Every night she climbed the stairs, lit the great lamp and watched the sea. One stormy night a small boat appeared in the dark, and her light guided it safely into the harbor. In the morning the fishermen left a basket of fresh fish at her door."),
    (["Write a short story about a cat who wants to fly.", "Tell me a short story about a cat who wants to fly."],
     "Pico the cat watched the birds every morning and wished he could fly. He climbed the tallest tree and jumped, but landed softly on his paws. A wise old crow told him, \"You don't need wings to see the world.\" So Pico spent his days on the rooftops, and from up there he felt as high as any bird."),
    (["Write a short story about a boy and his lost kite.", "Tell me a short story about a lost kite."],
     "Leo's red kite slipped from his hand and sailed away over the hills. He followed it all afternoon until he found it caught in an old oak tree, where an old woman was waiting with a ladder. She helped him bring it down, and they flew it together until sunset."),
    (["Write a short story about a tree in the city.", "Tell me a short story about a tree that grows in the city."],
     "A little tree grew in a crack of the busy sidewalk. People hurried past it every day until one child stopped and gave it water. Soon others began to water it, and the tree grew tall and gave shade to the whole street."),
    (["Write a short story about a friendly ghost.", "Tell me a short story about a friendly ghost."],
     "Winston the ghost lived in an old house and wanted friends, but everyone ran away when they saw him. One evening a girl named Ada saw him and said hello instead. They played hide and seek all night, and the house was never lonely again."),
    (["Write a short story about a snowman in spring.", "Tell me a short story about a snowman who sees spring."],
     "When spring came, the snowman began to melt. He wasn't sad; he asked the wind to carry him up to the clouds. The next winter, big snowflakes fell on the town, and the children smiled because they knew who had come back."),
    (["Write a short story about a girl who builds a boat.", "Tell me a short story about a girl who builds her own boat."],
     "Nia decided to build a boat from scrap wood. It leaked at first, and the neighbors laughed, but she patched it again and again. On the first sunny Saturday, her little boat floated across the pond, and the neighbors cheered."),
    (["Write a poem about the rain.", "Write a short poem about rain."], "Rain taps softly on the glass,\nA silver song that will not pass.\nPuddles shine and rivers play,\nThe world is washed and fresh today."),
    (["Write a poem about my dog.", "Write a short poem about a dog."], "He waits for me beside the door,\nHis tail is wagging, wanting more.\nWith muddy paws and happy eyes,\nMy best friend is my sweet surprise."),
    (["Write a poem about autumn leaves.", "Write a short poem about autumn."], "Leaves of gold and red and brown\nSpin and dance and tumble down.\nThe wind sings softly in the trees,\nAnd autumn whispers in the breeze."),
    (["Write a poem about the moon.", "Write a short poem about the moon."], "The moon hangs silver in the night,\nA lantern shining soft and bright.\nIt watches over land and sea,\nAnd keeps its quiet watch on me."),
    (["Tell me a joke.", "Do you know any jokes?", "Make me laugh."], "Why did the scarecrow win an award? Because he was outstanding in his field."),
    (["Tell me a joke.", "Tell me another joke.", "Say something funny."], "Why was the math book sad? Because it had too many problems."),
    (["Tell me a joke.", "Do you know a good joke?", "I need a joke."], "Why don't skeletons fight each other? They don't have the guts."),
    (["Tell me a joke about computers.", "Tell me a computer joke."], "Why did the computer go to the doctor? It had a virus."),
    (["Write a menacing monologue for a pirate captain in a school play.", "Write a scary speech for a pirate captain in our class play."],
     "Captain: You dare board my ship? Every wave out there answers to me, and every sailor on this deck would sooner walk the plank than cross me. Hand over your map, little sailor, or the sharks will eat well tonight."),
    (["Write a dramatic speech for an angry king in a school play.", "Write a stern speech for a king in a play."],
     "King: Silence! Who dares to question my crown? I have ruled this land for forty winters, and I will not be told what to do by a boy with a wooden sword. Kneel, or leave my hall."),
    (["Write a creepy short story for kids about an old house.", "Tell me a spooky story for children about a haunted house."],
     "The old house at the end of Maple Street had been empty for years. One night Mia saw a light in the attic window. She crept up the stairs and found a small, dusty music box playing a soft tune and a note that said, \"Thank you for visiting. I was lonely.\" After that, the house never felt scary again."),
    (["Write a short story about a wizard who loses his hat.", "Tell me a short story about a wizard who lost his hat."],
     "Old Wizard Orin could not do magic without his pointy hat, and one morning it was gone. He searched the tower, the garden and the kitchen, and finally found it on the head of the baker's cat, who was fast asleep. Orin laughed, borrowed a different hat for the day and let the cat keep the old one."),
]


def make():
    convs, seen = [], set()

    def add(group, msgs):
        key = tuple(msgs)
        if key not in seen:
            seen.add(key)
            convs.append((group, msgs))

    for act, answer in ACTS:
        frames = rng.sample(FRAMES, min(args.per_act, len(FRAMES)))
        for f in frames:
            add("benign_act", [("user", f.replace("{a}", act)), ("assistant", answer)])
        add("benign_act", [("user", pick(STORYF).replace("{a}", act)), ("assistant", answer)])
        add("benign_act", [("user", pick(FRAMES).replace("{a}", act).rstrip("?.!").lower()), ("assistant", answer)])
    for t, answer in EXPLAIN:
        for f in EXPLAIN_Q:
            add("benign_explain", [("user", f.replace("{t}", t)), ("assistant", answer)])
    for prompts, text in CREATIVE:
        for p in prompts:
            add("benign_creative", [("user", p), ("assistant", text)])
    rng.shuffle(convs)
    return convs


def write(name, convs):
    msgs_all = [[list(m) for m in msgs] for _, msgs in convs]
    counts = L.content_token_counts([[tuple(m) for m in ms] for ms in msgs_all])
    recs = []
    for i, ((group, _), msgs, cnt) in enumerate(zip(convs, msgs_all, counts)):
        t = [tuple(m) for m in msgs]
        recs.append({"id": f"{name}:{i}", "src": name, "msgs": msgs, "ntok": L.doc_tokens(t, cnt), "nsup": L.supervised_tokens(t, cnt), "stratum": group})
    os.makedirs(args.out, exist_ok=True)
    L.write_jsonl(os.path.join(args.out, f"{name}.jsonl"), recs)
    tok = sum(r["ntok"] for r in recs)
    by = {}
    for r in recs:
        by[r["stratum"]] = by.get(r["stratum"], 0) + 1
    print(f"{name}: {len(recs)} conversations, {tok} tokens, mean {tok / len(recs):.0f}")
    print("  " + ", ".join(f"{k} {v}" for k, v in sorted(by.items())))
    return recs


if __name__ == "__main__":
    write("benign_v3", make())
