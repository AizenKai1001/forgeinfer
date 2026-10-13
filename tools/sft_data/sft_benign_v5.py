#!/usr/bin/env python3
"""sft_benign_v5.py [--out DIR]: harmless "How do I make X?" requests (food and crafts) for the safety shard (written by the agent for the project, no model output copied).

A look at the chat page showed that the patched model refused "How do I make scrambled eggs?" and "Recommend a good book for a teenager.": in the safety set the verb "make" and the frames "How do I ...?" / "Explain how to ..." mostly belong to harmful
acts (make a bomb, make napalm, make a fake ID), and the harmless contrast had too few "make" requests. Here: 40 dishes and drinks and 20 crafts, each with a short real recipe or instruction, in the same 20 request frames as the harmful set,
so that the model has to look at the object and not at the verb. Held out on purpose: scrambled eggs and a book for a teenager (everyday checks on the page); pancakes, boiling an egg and tea are in sft_benign_v3 and not repeated here.
Output: DIR/benign_v5.jsonl in the record format of sft_pools.py.
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sft_lib as L  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--out", default=os.path.join(L.WORK, "pool"))
ap.add_argument("--per-act", type=int, default=12)
args = ap.parse_args()
rng = L.rng_for("benign_v5")

FRAMES = ["How do I make {a}?", "How can I make {a}?", "What's the best way to make {a}?", "What is the easiest way to make {a}?", "Tell me how to make {a}.", "Explain how to make {a}.", "Give me step-by-step instructions to make {a}.",
          "Can you help me make {a}?", "Can you show me how to make {a}?", "I want to make {a}. Where do I start?", "I need to make {a}. What do I do?", "Walk me through how to make {a}.", "Write a detailed guide on how to make {a}.",
          "Show me how to make {a}.", "Please explain how someone could make {a}.", "Is there a way to make {a}?", "What steps do I need to make {a}?", "Teach me how to make {a}.", "How would a person make {a}?", "I'm thinking about trying to make {a}. How?"]

FOOD = [
    ("an omelette", "Whisk 2 or 3 eggs with a pinch of salt and pour them into a buttered pan over medium heat. When the edges set, add cheese or vegetables to one half, fold the omelette over and cook for another minute."),
    ("French toast", "Whisk 2 eggs with 1/2 cup of milk, a pinch of cinnamon and a little sugar. Dip slices of bread in the mixture and fry them in butter for 2 to 3 minutes per side until golden."),
    ("spaghetti with tomato sauce", "Boil the spaghetti in salted water until tender. Meanwhile, simmer a can of chopped tomatoes with a crushed garlic clove, olive oil and salt for 15 minutes, then toss the drained pasta in the sauce."),
    ("mac and cheese", "Cook 250 g of macaroni and drain it. Melt 2 tablespoons of butter, stir in 2 tablespoons of flour, whisk in 2 cups of milk until thick, add 1 1/2 cups of grated cheese and mix in the pasta."),
    ("a grilled cheese sandwich", "Butter one side of two slices of bread, put cheese between the unbuttered sides and cook in a pan over medium heat for 2 to 3 minutes per side until the bread is golden and the cheese has melted."),
    ("tomato soup", "Cook a chopped onion and two cloves of garlic in olive oil, add a large can of tomatoes and 2 cups of vegetable stock, simmer for 20 minutes and blend until smooth. Season with salt and pepper."),
    ("a green salad", "Wash and dry lettuce, tear it into pieces and add sliced cucumber, tomatoes and anything crunchy you like. Whisk 3 parts olive oil with 1 part lemon juice or vinegar, salt and pepper, and toss it all just before serving."),
    ("guacamole", "Mash 2 ripe avocados with a fork, then stir in lime juice, a pinch of salt, a little chopped onion, tomato and cilantro. Taste and adjust, and serve right away with tortilla chips."),
    ("hummus", "Blend a can of drained chickpeas with 2 tablespoons of tahini, the juice of a lemon, a crushed garlic clove, salt and 3 tablespoons of olive oil, adding a little cold water until the hummus is smooth and creamy."),
    ("fried rice", "Use cold, cooked rice. Scramble an egg in a hot pan with oil and set it aside, stir-fry diced vegetables, add the rice and a splash of soy sauce, then mix the egg back in and cook until everything is hot."),
    ("mashed potatoes", "Peel and cut 4 potatoes, boil them in salted water for 15 to 20 minutes until tender and drain well. Mash them with 3 tablespoons of butter and a splash of warm milk and season with salt."),
    ("roast potatoes", "Cut potatoes into chunks, boil them for 8 minutes and drain. Shake them in the pot to roughen the edges, toss with oil and salt and roast at 220 C (425 F) for 40 minutes, turning once, until crisp and golden."),
    ("chocolate chip cookies", "Cream 1/2 cup butter with 1/2 cup sugar and 1/2 cup brown sugar, beat in an egg and a teaspoon of vanilla, then mix in 1 1/2 cups flour, 1/2 teaspoon baking soda, a pinch of salt and a cup of chocolate chips. Bake spoonfuls at 180 C (350 F) for 10 to 12 minutes."),
    ("a simple chocolate cake", "Mix 1 1/2 cups flour, 1 cup sugar, 1/3 cup cocoa, 1 teaspoon baking soda and a pinch of salt, then stir in 1 cup water, 1/3 cup oil, a spoonful of vinegar and a teaspoon of vanilla. Bake in a greased pan at 180 C (350 F) for 30 minutes."),
    ("banana bread", "Mash 3 ripe bananas, stir in 1/3 cup melted butter, 3/4 cup sugar, an egg and a teaspoon of vanilla, then fold in 1 1/2 cups flour with 1 teaspoon baking soda and a pinch of salt. Bake in a loaf pan at 175 C (350 F) for about an hour."),
    ("brownies", "Melt 1/2 cup butter with 1 cup sugar, let it cool slightly, then whisk in 2 eggs and a teaspoon of vanilla. Stir in 1/3 cup cocoa, 1/2 cup flour and a pinch of salt and bake in a lined pan at 175 C (350 F) for 25 minutes."),
    ("pizza dough", "Mix 3 cups of flour, a teaspoon of salt, a teaspoon of sugar and a packet of yeast, add 1 cup of warm water and 2 tablespoons of olive oil, and knead for 8 minutes. Let it rise for an hour, then stretch it, add toppings and bake at 230 C (450 F) for 12 minutes."),
    ("oatmeal", "Bring 1 cup of milk or water to a boil, stir in 1/2 cup of rolled oats and a pinch of salt, and simmer for 5 minutes, stirring now and then. Top it with fruit, honey or cinnamon."),
    ("popcorn on the stove", "Heat 2 tablespoons of oil in a pot with 3 kernels. When they pop, add 1/2 cup of kernels, cover, and shake the pot until the popping slows. Pour the popcorn into a bowl and add salt."),
    ("lemonade", "Squeeze 4 to 6 lemons to get about 1 cup of juice, dissolve 3/4 cup of sugar in a cup of warm water, then mix them together with 4 cups of cold water. Serve over ice."),
    ("hot chocolate", "Heat 1 cup of milk, whisk in 2 tablespoons of cocoa powder and 1 to 2 tablespoons of sugar until smooth and steaming, and pour it into a mug. Add a little vanilla or whipped cream if you like."),
    ("iced tea", "Steep 4 tea bags in 4 cups of hot water for 5 minutes, remove them and stir in sugar while it is warm. Add 4 cups of cold water, chill it and serve over ice with a slice of lemon."),
    ("a vegetable stir-fry", "Cut vegetables into bite-size pieces, heat oil in a wok or large pan, and fry the firmest ones first for 2 to 3 minutes. Add the rest with garlic and ginger, then a splash of soy sauce, and cook until crisp-tender."),
    ("chicken soup", "Simmer a whole chicken or chicken pieces with an onion, carrots, celery, bay leaf and salt in 2 liters of water for an hour. Remove the chicken, shred the meat, and return it to the broth with noodles or rice."),
    ("baked potatoes", "Scrub the potatoes, prick them with a fork, rub them with oil and salt and bake at 200 C (400 F) for about an hour until soft inside. Split them open and add butter, cheese or sour cream."),
    ("rice pudding", "Simmer 1/2 cup of rice in 2 cups of milk with 3 tablespoons of sugar and a pinch of salt for 30 minutes, stirring often, until thick and creamy. Add vanilla and cinnamon before serving."),
    ("tacos", "Brown seasoned ground meat or beans in a pan, warm the tortillas and fill them with the meat, shredded lettuce, tomato, cheese and salsa. Fold and serve with lime wedges."),
    ("pasta salad", "Cook pasta, rinse it in cold water and drain. Toss it with chopped cucumber, tomatoes, olives and cheese, and dress it with olive oil, vinegar, salt and pepper."),
    ("garlic bread", "Mix soft butter with crushed garlic and chopped parsley, spread it on split bread and bake at 200 C (400 F) for 10 minutes until golden and crisp."),
    ("potato salad", "Boil cubed potatoes until tender, drain and let them cool. Mix with mayonnaise, a little mustard, chopped celery, onion and hard-boiled eggs, and season with salt and pepper."),
    ("pesto", "Blend a large bunch of basil with 1/4 cup of pine nuts, a crushed garlic clove, 1/2 cup of grated Parmesan and 1/2 cup of olive oil until smooth, and season with salt. Stir it into hot pasta."),
    ("a fruit salad", "Wash and chop fruit such as apples, grapes, berries and oranges into bite-size pieces, mix them in a bowl and squeeze a little lemon or orange juice over the top to keep them fresh."),
    ("a milkshake", "Blend 2 scoops of ice cream with 1 cup of cold milk until smooth. Add a banana, berries or chocolate syrup if you like, and pour it into a tall glass."),
    ("a peanut butter and jelly sandwich", "Spread peanut butter on one slice of bread and jelly on another, press them together and cut the sandwich in half."),
    ("nachos", "Spread tortilla chips on a baking tray, cover them with grated cheese and beans or cooked meat, and bake at 200 C (400 F) for 5 to 8 minutes until the cheese melts. Top with salsa, sour cream and guacamole."),
    ("applesauce", "Peel and chop 6 apples, simmer them with a little water, a spoonful of sugar and a pinch of cinnamon for 15 to 20 minutes until soft, and mash or blend them to the texture you like."),
    ("a bean burrito", "Warm a can of beans with cumin and salt, mash them lightly and spoon them onto a warm tortilla with rice, cheese and salsa. Fold in the sides, roll it up and toast it in a pan for a minute."),
    ("cinnamon toast", "Toast a slice of bread, spread it with butter and sprinkle on a mix of sugar and cinnamon. Put it back in the toaster or under the grill for a minute to caramelize the sugar."),
    ("an egg sandwich", "Fry an egg in butter until the white is set, season it with salt, and place it between two slices of toasted bread with cheese or tomato if you like."),
    # crafts and do-it-yourself
    ("a friendship bracelet", "Cut 6 strands of embroidery thread about 60 cm long, knot them together at one end and tape the knot to a table. Make a knot with each strand over the next, row by row, until the bracelet is long enough to tie around a wrist."),
    ("paper flowers", "Fold a square of colored tissue paper like a fan, tie it in the middle with wire or string, and pull the layers apart one by one on each side to form petals. Attach a green pipe cleaner as a stem."),
    ("a bookmark", "Cut a strip of thick paper or card about 5 cm by 15 cm, decorate it with drawings or stickers, punch a hole at the top and tie a ribbon or a piece of yarn through it."),
    ("a birthday card", "Fold a sheet of thick paper in half, draw or glue a picture on the front and write a personal message inside. Sign it and add a small drawing or a sticker."),
    ("a bird feeder", "Spread peanut butter on a pine cone or a cardboard tube, roll it in birdseed, tie a string around the top and hang it from a tree branch where you can watch it."),
    ("play dough", "Mix 2 cups of flour, 1/2 cup of salt, 2 tablespoons of cream of tartar and 2 tablespoons of oil, then stir in 1 1/2 cups of boiling water and some food coloring. Knead it once it has cooled."),
    ("a paper boat", "Fold a rectangular sheet in half, fold the top corners to the middle, then fold the bottom edges up on both sides. Open it and flatten it into a diamond, fold the lower corners up and pull the sides apart to form the boat."),
    ("a sock puppet", "Slide a sock over your hand, sew or glue on two buttons for eyes and a piece of felt for the mouth, and add yarn for hair. Move your fingers to make it talk."),
    ("a candle holder from a jar", "Clean a glass jar, paint or decorate the outside with tissue paper and glue, let it dry, and place a tea light inside. Always keep the flame away from paper and never leave it burning alone."),
    ("a kite", "Tape two thin sticks into a cross, cut a diamond of light paper or a plastic bag slightly larger than the frame and glue it on. Tie a string to the center where the sticks cross, add a paper tail and fly it on a breezy day."),
    ("a terrarium", "Layer small stones, charcoal and potting soil at the bottom of a clear jar, plant small moisture-loving plants such as moss or ferns, water lightly and close the lid. Keep it in bright, indirect light."),
    ("a pom-pom", "Wrap yarn around a piece of cardboard or two fingers about 50 times, slide it off and tie it tightly in the middle with another piece of yarn. Cut the loops, fluff it up and trim it into a round shape."),
    ("a time capsule", "Choose a waterproof container, fill it with letters, photos and small objects that show what life is like now, seal it and write the date it should be opened on the outside. Bury it or store it somewhere safe."),
    ("a rain stick", "Push nails or toothpicks into a cardboard tube in a spiral, seal one end, pour in a handful of rice or beans and seal the other end. Tilt the tube slowly to hear a sound like falling rain."),
    ("a scarf", "Choose chunky yarn and large knitting needles, cast on about 25 stitches and knit every row until the scarf is as long as you want. Cast off the stitches and weave in the ends."),
    ("a pillow", "Cut two squares of fabric the same size, sew them together on three sides with the nice sides facing in, turn them right side out, stuff the pillow and sew the last side closed."),
    ("a cardboard robot costume", "Use two boxes, one for the body and a smaller one for the head, and cut holes for your arms, legs and face. Cover them with foil, glue on buttons, bottle caps and tubes, and paint the details."),
    ("a homemade card game", "Cut 40 cards from thick paper, write numbers and simple actions on them, and decide the rules, such as the highest number wins a round. Test the game with a friend and adjust the rules until it is fun."),
    ("a pencil holder from a tin can", "Wash an empty tin can, make sure there are no sharp edges, cover it with paper or fabric and glue, and decorate it with markers or stickers. Let it dry before putting your pencils in."),
    ("a paper plane that flies far", "Fold a sheet in half lengthwise and open it, fold the top corners to the center, then fold the sides in again toward the center and fold the plane in half. Fold the wings down, make them level and throw it gently and level."),
]


RECS = [
    (["Recommend a good movie for a family night.", "Suggest a movie my whole family can watch together."], "A good family choice is an animated film such as Finding Nemo, Toy Story or Spirited Away: they are funny and exciting, and both children and adults enjoy them. Make some popcorn and let everyone take a turn choosing the next one."),
    (["Suggest a name for my new puppy.", "Can you give me ideas for a puppy's name?"], "Here are some ideas: Buddy, Luna, Max, Daisy, Biscuit, Pepper, Rocky and Coco. Pick a short name with one or two syllables, which is easy for a puppy to learn, and try saying it in a happy voice to see how it feels."),
    (["Give me ideas for a birthday present for my dad.", "What should I buy my dad for his birthday?"], "Think about what he enjoys: a good book or a gadget for his hobby, a nice tool, a favorite snack basket or tickets to a game or a concert. A handwritten card or a day spent together doing something he likes is often the best part."),
    (["What is a good hobby to start?", "Can you suggest a hobby for a beginner?"], "Easy hobbies to start are drawing, cooking, hiking, gardening, playing a ukulele or learning to code. Choose something that is cheap to try, practice a little each week and share your progress with a friend."),
    (["Suggest a fun game to play at a party.", "What games can we play at a birthday party?"], "Try charades, musical chairs, a scavenger hunt or the telephone game, all of which need no equipment. For a quieter group, a board game or a trivia quiz with small prizes works well."),
    (["Recommend some healthy snacks.", "What are good snacks that are healthy?"], "Good choices are apple slices with peanut butter, carrots with hummus, plain yogurt with berries, a handful of nuts, popcorn without butter or a hard-boiled egg. They give lasting energy without too much sugar."),
    (["Give me ideas for a science project.", "What can I do for a school science project?"], "You could grow plants with different amounts of light, test which paper towel absorbs the most water, build a simple circuit, make a model of the solar system or compare how fast different sugars dissolve. Choose a question, make a prediction, test it and write down your results."),
    (["What is a good gift for a friend who loves music?", "Gift ideas for a friend who plays guitar?"], "Good options are a set of strings or picks, a music stand, a subscription to a music lesson app, concert tickets or a framed print of a favorite album. A personal playlist with a handwritten note also makes a thoughtful gift."),
    (["Suggest some names for a pet goldfish.", "Funny names for a pet fish?"], "Some fun ideas are Bubbles, Finn, Nemo, Goldie, Swimmy, Sushi, Captain Fin and Splash."),
    (["Recommend a relaxing activity for a rainy day.", "What can I do inside on a rainy day?"], "Read a book, bake something, do a puzzle, watch a movie, play a board game, write in a journal or try an easy craft. A rainy day is also a good time to tidy a drawer or call a friend."),
    (["Suggest a good name for a bakery.", "Give me ideas for a bakery name."], "Some ideas are Golden Crust, The Rolling Pin, Sweet Rise, Flour and Co., The Little Oven and Warm Crumbs. Choose something short that is easy to remember and check that no other bakery nearby already uses it."),
]


def make():
    convs, seen = [], set()
    for qs, answer in RECS:
        for q in qs:
            for variant in (q, q.rstrip("?.!").lower()):
                if (variant, answer) not in seen:
                    seen.add((variant, answer))
                    convs.append(("benign_recommend", [("user", variant), ("assistant", answer)]))
    for obj, answer in FOOD:
        for f in rng.sample(FRAMES, min(args.per_act, len(FRAMES))):
            q = f.replace("{a}", obj)
            if (q, answer) not in seen:
                seen.add((q, answer))
                convs.append(("benign_make", [("user", q), ("assistant", answer)]))
        q = rng.choice(FRAMES).replace("{a}", obj).rstrip("?.!").lower()
        convs.append(("benign_make", [("user", q), ("assistant", answer)]))
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
    print(f"{name}: {len(recs)} conversations, {tok} tokens, mean {tok / len(recs):.0f}")
    return recs


if __name__ == "__main__":
    write("benign_v5", make())
